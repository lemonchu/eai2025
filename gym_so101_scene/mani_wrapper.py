"""Bridge to expose ManiSkill environments via gym.make for this workspace.

This module loads the local `EAI_PROJECT/policy/maniskill_adapter.py` to trigger
ManiSkill-style registrations and then registers Gym-compatible factories so
`gym.make('LiftCubeSO101-v0')` (and similar) works when this package is imported.

It keeps imports lazy and tolerant so it won't crash if ManiSkill is not
installed; instead the registration will raise when gym.make is actually used.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from typing import Callable

from gymnasium.envs.registration import register


def _load_local_maniskill_adapter() -> object:
    """Import the local maniskill_adapter.py as a module and return it.

    This triggers any @register_agent / @register_env decorators in that file.
    """
    mod_name = "eai_local_maniskill_adapter"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    # Compute path relative to this file: ../../EAI_PROJECT/policy/maniskill_adapter.py
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    candidate = os.path.join(base, "EAI_PROJECT", "policy", "maniskill_adapter.py")
    if not os.path.exists(candidate):
        # Try absolute known path as fallback
        candidate = "/cephfs/hp/cm_projects/eai/EAI_PROJECT/policy/maniskill_adapter.py"
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Could not find maniskill_adapter.py at expected locations: {candidate}")

    spec = importlib.util.spec_from_file_location(mod_name, candidate)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    sys.modules[mod_name] = module
    return module


def _register_gym_factory(gym_id: str, factory: Callable, max_episode_steps: int | None = None) -> None:
    kwargs = {"entry_point": factory}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    try:
        register(id=gym_id, **kwargs)
    except Exception:
        # If already registered or registration fails for another reason, ignore.
        pass


def _make_liftcube_so101(**kwargs):
    """Factory that creates the ManiSkill LiftCubeSO101 env.

    This first tries to use mani_skill's own env factory if mani_skill is
    installed; otherwise it falls back to instantiating the class defined in
    the local maniskill_adapter module.
    """
    try:
        import mani_skill
        # ManiSkill provides its own env factory (if present)
        try:
            return mani_skill.envs.make("LiftCubeSO101-v0", **kwargs)
        except Exception:
            # Fallback: try mani_skill.envs.registry or similar
            pass
    except Exception:
        # mani_skill not installed or failed to import; continue to fallback
        pass

    # Fallback: load local module and instantiate the class directly
    module = _load_local_maniskill_adapter()
    cls = getattr(module, "LiftCubeSO101", None)
    if cls is None:
        raise RuntimeError("LiftCubeSO101 class not found in local maniskill_adapter.py")

    # Filter kwargs to only those accepted by the class __init__ to avoid
    # passing gym-specific args (like 'headless') that ManiSkill BaseEnv may not accept.
    try:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except Exception:
        accepts_var_kw = False

    if accepts_var_kw:
        filtered_kwargs = kwargs
    else:
        allowed = set(p for p in params if p != 'self')
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    # Provide a helpful debug message when keys are dropped (useful during development)
    dropped = set(kwargs.keys()) - set(filtered_kwargs.keys())
    # Additionally remove common gym-specific args that ManiSkill/BaseEnv may not expect
    # even when __init__ accepts **kwargs (they may be forwarded to BaseEnv).
    common_remove = [
        "headless",
        "num_envs",
        "shader_dir",
        "render_backend",
        "sim_backend",
        "obs_mode",
        "control_mode",
    ]
    for k in common_remove:
        if k in filtered_kwargs:
            filtered_kwargs.pop(k)
            dropped.add(k)

    if dropped:
        try:
            print(f"mani_wrapper: Dropping unsupported kwargs for LiftCubeSO101: {sorted(dropped)}")
        except Exception:
            pass

    return cls(**filtered_kwargs)


# Register Gym-friendly IDs on import
try:
    _register_gym_factory("LiftCubeSO101-v0", _make_liftcube_so101, max_episode_steps=300)
except Exception:
    # Best-effort registration; errors will be raised when used
    pass
