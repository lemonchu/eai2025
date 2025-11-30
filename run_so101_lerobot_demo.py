#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
# Local assets live under eai2025/scene/assets/SO101 as provided by the TAs.
DEFAULT_SCENE_ROOT = PROJECT_ROOT / "scene" / "assets" / "SO101"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gym_so101_scene  # noqa: F401


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
    record_root = Path(args.record_dir).resolve()
    record_root.mkdir(parents=True, exist_ok=True)
    for episode in range(args.episodes):
        seed = args.seed + episode if args.seed is not None else None
        obs, info = env.reset(seed=seed)
        done = False
        step_idx = 0
        episode_dir = record_root / f"episode_{episode:03d}"
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            for camera_name, frame in obs["images"].items():
                save_rgb(frame, episode_dir / camera_name / f"frame_{step_idx:04d}.png")
            step_idx += 1
            done = terminated or truncated
        print(
            f"Episode {episode} -> success={info.get('success', False)} "
            f"distance={info.get('distance', np.nan):.3f}"
        )
    env.close()
    example = (
        "ffmpeg -y -framerate 30 -pattern_type glob -i '"
        f"{record_root}/episode_000/front/*.png' {record_root}/episode_000_front.mp4"
    )
    print("Frames saved under", record_root, "\nExample:", example)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SO-101 Sapien scene using local assets")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes to roll out")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--record-dir", type=str, default="outputs/so101_demo", help="Directory for saved frames")
    parser.add_argument("--scene-root", type=str, default=None, help="Override path to SO-101 assets")
    parser.add_argument("--task", type=str, default="lift", help="Task to spawn: lift|stack|sort")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument("--show-viewer", action="store_true", help="Open the Sapien viewer (requires GUI)")
    return parser.parse_args()


if __name__ == "__main__":
    rollout(parse_args())
