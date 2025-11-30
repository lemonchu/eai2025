#!/usr/bin/env python3
"""Reference-style policy runner that mirrors the ManiSkill baseline structure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_ROOT = PROJECT_ROOT / "scene" / "assets" / "SO101"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gym_so101_scene  # noqa: F401

try:
    from .ik_pick_policy import IKPickPolicy
except ImportError:  # pragma: no cover
    from policies.reference_pick.ik_pick_policy import IKPickPolicy


def save_rgb(frame: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


def _resolve_scene_root(scene_root: str | None) -> Path:
    if scene_root:
        root = Path(scene_root).expanduser().resolve()
    else:
        root = DEFAULT_SCENE_ROOT
    if not root.exists():
        raise FileNotFoundError(f"SO-101 assets not found at {root}")
    return root


def rollout(args: argparse.Namespace) -> None:
    scene_root = _resolve_scene_root(args.scene_root)
    env = gym.make(
        "gym_so101_scene/ScenePickCube-v0",
        task=args.task,
        scene_root=str(scene_root),
        render_mode="rgb_array",
        headless=not args.show_viewer,
        max_episode_steps=args.max_steps,
    )
    policy = IKPickPolicy(env, arm_name=args.arm)
    record_root = Path(args.record_dir).resolve()
    record_root.mkdir(parents=True, exist_ok=True)

    for episode in range(args.episodes):
        seed = args.seed + episode if args.seed is not None else None
        obs, info = env.reset(seed=seed)
        policy.reset()
        done = False
        step_idx = 0
        episode_dir = record_root / f"episode_{episode:03d}"
        print(f"Episode {episode}: seed={seed}")
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = obs["images"].get("front")
            if frame is not None:
                save_rgb(frame, episode_dir / "front" / f"frame_{step_idx:04d}.png")
            # Save top image: prefer obs images if present, else fallback to take_picture
            try:
                top_img = obs["images"].get("top")
                if top_img is None:
                    top_img = env.take_picture("top")
                if top_img is not None:
                    save_rgb(top_img, episode_dir / "top" / f"frame_{step_idx:04d}.png")
            except Exception:
                pass

            # Save side image
            try:
                side_img = obs["images"].get("side")
                if side_img is None:
                    side_img = env.take_picture("side")
                if side_img is not None:
                    save_rgb(side_img, episode_dir / "side" / f"frame_{step_idx:04d}.png")
            except Exception:
                pass

            # Save right-side image (now included in obs if env supports it)
            try:
                right_img = obs["images"].get("right_side")
                if right_img is None:
                    # fallback to env.take_picture
                    try:
                        right_img = env.take_picture("right_side")
                    except Exception:
                        right_img = None
                if right_img is not None:
                    save_rgb(right_img, episode_dir / "right_side" / f"frame_{step_idx:04d}.png")
            except Exception:
                pass
            if step_idx % 1 == 0:
                lift = info.get("max_lift", np.nan)
                distance = info.get("distance", np.nan)
                gripped = info.get("gripped", False)
                print(
                    f"  step={step_idx:<4} stage={policy.current_stage:<8} reward={reward: .3f} dist={distance: .3f} lift={lift: .3f} m gripped={bool(gripped)}"
                )
            done = terminated or truncated
            step_idx += 1
        print(
            f"Episode {episode} -> success={info.get('success', False)} max_lift={info.get('max_lift', np.nan):.3f} m gripped={info.get('gripped', False)} steps={step_idx}"
        )
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IKPickPolicy against the SO-101 scene")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to roll out")
    parser.add_argument("--max-steps", type=int, default=220, help="Max steps per episode")
    parser.add_argument("--record-dir", type=str, default="outputs/reference_pick", help="Directory for saved frames")
    parser.add_argument("--scene-root", type=str, default=None, help="Override path to SO-101 assets")
    parser.add_argument("--task", type=str, default="pick_cube", help="Task to spawn: lift|stack|sort|pick_cube")
    parser.add_argument("--arm", type=str, default="right", help="Arm to control (left|right)")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument("--show-viewer", action="store_true", help="Open the Sapien viewer (requires GUI)")
    return parser.parse_args()


if __name__ == "__main__":
    rollout(parse_args())
