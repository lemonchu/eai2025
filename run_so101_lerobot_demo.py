#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = PROJECT_ROOT.parent
LEROBOT_SRC = PARENT_ROOT / "lerobot" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gym_so101_scene  # noqa: F401
from lerobot.envs.factory import make_env_config


def save_rgb(frame: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


def rollout(args: argparse.Namespace) -> None:
    scene_root = Path(args.scene_root).resolve() if args.scene_root else None
    cfg = make_env_config(
        "so101_scene",
        task="ScenePickCube-v0",
        scene_root=str(scene_root) if scene_root else None,
        render_mode="rgb_array",
        headless=not args.show_viewer,
        max_episode_steps=args.max_steps,
    )
    env = gym.make(cfg.gym_id, **cfg.gym_kwargs)
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
            front = obs["images"]["front"]
            wrist = obs["images"]["wrist"]
            save_rgb(front, episode_dir / "front" / f"frame_{step_idx:04d}.png")
            save_rgb(wrist, episode_dir / "wrist" / f"frame_{step_idx:04d}.png")
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
    parser = argparse.ArgumentParser(description="Run the SO-101 Sapien scene via LeRobot config")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes to roll out")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--record-dir", type=str, default="outputs/so101_demo", help="Directory for saved frames")
    parser.add_argument("--scene-root", type=str, default=None, help="Override path to SO-101 assets")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument("--show-viewer", action="store_true", help="Open the Sapien viewer (requires GUI)")
    return parser.parse_args()


if __name__ == "__main__":
    rollout(parse_args())
