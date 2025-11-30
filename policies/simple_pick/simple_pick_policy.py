from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class StageTarget:
    name: str
    joint_targets: np.ndarray


class SimplePickPolicy:
    """Very small finite-state policy that executes a scripted pick motion.

    The policy operates directly in joint space using the environment's
    delta-joint action interface. It assumes the "right" arm is used,
    but the arm name can be overridden via the constructor.
    """

    def __init__(
        self,
        arm_name: str,
        arm_index: int,
        dof_per_arm: int,
        total_dofs: int,
        action_scale: float,
        stage_targets: Sequence[StageTarget] | None = None,
        tolerance: float = 0.03,
    ) -> None:
        self.arm_name = arm_name
        self.arm_index = arm_index
        self.dof_per_arm = dof_per_arm
        self.total_dofs = total_dofs
        self.action_scale = action_scale
        self.tolerance = tolerance
        self._stage_idx = 0
        self._cached_zero: np.ndarray | None = None
        if stage_targets is None:
            stage_targets = self._default_stage_targets()
        self._stages = list(stage_targets)

    @classmethod
    def from_env(
        cls,
        env,
        arm_name: str = "right",
        action_scale: float | None = None,
        tolerance: float = 0.03,
    ) -> "SimplePickPolicy":
        unwrapped = env.unwrapped
        dof = len(getattr(unwrapped, "_controlled_joint_names"))
        total_dofs = env.action_space.shape[0]
        num_arms = total_dofs // dof
        # Map arm name to index (env stores ArmInterface objects with .name)
        arm_lookup = {arm.name: idx for idx, arm in enumerate(getattr(unwrapped, "_arms", []))}
        if arm_name not in arm_lookup:
            raise ValueError(f"Arm '{arm_name}' not found in env arms: {list(arm_lookup)}")
        scale = action_scale if action_scale is not None else getattr(unwrapped, "_action_scale", 0.05)
        return cls(
            arm_name=arm_name,
            arm_index=arm_lookup[arm_name],
            dof_per_arm=dof,
            total_dofs=total_dofs,
            action_scale=scale,
            stage_targets=None,
            tolerance=tolerance,
        )

    def reset(self) -> None:
        self._stage_idx = 0

    @property
    def current_stage(self) -> str:
        if self._stage_idx >= len(self._stages):
            return "done"
        return self._stages[self._stage_idx].name

    def act(self, obs: dict) -> np.ndarray:
        if self._cached_zero is None:
            self._cached_zero = np.zeros(self.total_dofs, dtype=np.float32)
        action = np.zeros_like(self._cached_zero)
        if self._stage_idx >= len(self._stages):
            return action
        arm_state_key = f"{self.arm_name}_joint_pos"
        state_dict = obs.get("state")
        if state_dict is None or arm_state_key not in state_dict:
            raise KeyError(f"Observation missing '{arm_state_key}'")
        joints = np.asarray(state_dict[arm_state_key], dtype=np.float32)
        stage = self._stages[self._stage_idx]
        target = stage.joint_targets
        delta = np.clip(target - joints, -self.action_scale, self.action_scale)
        start = self.arm_index * self.dof_per_arm
        stop = start + self.dof_per_arm
        action[start:stop] = delta
        if np.linalg.norm(joints - target) < self.tolerance:
            self._stage_idx += 1
        return action

    def _default_stage_targets(self) -> list[StageTarget]:
        # Heuristic joint positions tuned for the default SO-101 layout.
        pregrasp = np.array([0.1, -0.9, 1.2, 0.7, 0.0, 0.6], dtype=np.float32)
        descend = np.array([0.1, -0.3, 0.8, 0.5, 0.0, 0.6], dtype=np.float32)
        grasp = descend.copy()
        grasp[-1] = -0.1  # close gripper
        lift = np.array([0.1, -1.1, 1.3, 0.9, 0.0, -0.1], dtype=np.float32)
        return [
            StageTarget("pregrasp", pregrasp),
            StageTarget("descend", descend),
            StageTarget("close", grasp),
            StageTarget("lift", lift),
        ]