#!/usr/bin/env python3
"""Minimal REINFORCE loop for the pick-cube task (headless-friendly)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_ROOT = PROJECT_ROOT / "scene" / "assets" / "SO101"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gym_so101_scene  # noqa: F401


class StateVectorWrapper(gym.ObservationWrapper):
    """Extract low-dimensional joint features for learning."""

    def __init__(self, env: gym.Env, arm: str = "right") -> None:
        super().__init__(env)
        self.arm = arm
        unwrapped = env.unwrapped
        self.dof = len(getattr(unwrapped, "_controlled_joint_names"))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.dof * 2,),
            dtype=np.float32,
        )

    def observation(self, obs):  # type: ignore[override]
        state = obs.get("state", {})
        pos = np.asarray(state.get(f"{self.arm}_joint_pos"), dtype=np.float32)
        vel = np.asarray(state.get(f"{self.arm}_joint_vel"), dtype=np.float32)
        if pos.size == 0:
            pos = np.zeros(self.dof, dtype=np.float32)
        if vel.size == 0:
            vel = np.zeros(self.dof, dtype=np.float32)
        return np.concatenate([pos, vel], dtype=np.float32)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.net(obs)
        std = torch.exp(self.log_std).clamp(min=1e-3, max=2.0)
        return mu, std

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        mu, std = self(obs_t)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.detach().cpu().numpy(), log_prob


def _resolve_scene_root(scene_root: str | None) -> Path:
    if scene_root:
        root = Path(scene_root).expanduser().resolve()
    else:
        root = DEFAULT_SCENE_ROOT
    if not root.exists():
        raise FileNotFoundError(f"SO-101 assets not found at {root}")
    return root


def train(args: argparse.Namespace) -> None:
    scene_root = _resolve_scene_root(args.scene_root)
    env = gym.make(
        "gym_so101_scene/ScenePickCube-v0",
        task="pick_cube",
        scene_root=str(scene_root),
        render_mode="rgb_array",
        headless=True,
        max_episode_steps=args.max_steps,
    )
    env = StateVectorWrapper(env, arm=args.arm)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = GaussianPolicy(obs_dim, act_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    rng = np.random.default_rng(args.seed)
    for episode in range(args.episodes):
        seed = int(rng.integers(0, 2**31 - 1)) if args.seed is not None else None
        obs, _ = env.reset(seed=seed)
        log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        total_reward = 0.0
        for step in range(args.max_steps):
            action, log_prob = policy.act(obs)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(action)
            log_probs.append(log_prob)
            rewards.append(float(reward))
            total_reward += float(reward)
            if terminated or truncated:
                break
        if log_probs:
            returns = _compute_returns(rewards, args.gamma)
            stacked = torch.stack(log_probs)
            loss = -(stacked * returns).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item = float(loss.item())
        else:
            loss_item = 0.0
        success = bool(info.get("success", False)) if rewards else False
        print(
            f"episode={episode:04d} steps={len(rewards):03d} reward={total_reward:7.3f} "
            f"loss={loss_item:.4f} success={success}"
        )
    if args.output:
        torch.save({"model_state": policy.state_dict()}, args.output)
        print("Saved policy to", args.output)
    env.close()


def _compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    tensor = torch.as_tensor(returns, dtype=torch.float32)
    if tensor.numel() > 1:
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
    return tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple policy on the pick-cube task")
    parser.add_argument("--episodes", type=int, default=30, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--arm", type=str, default="right", help="Arm to control")
    parser.add_argument("--scene-root", type=str, default=None, help="Override path to SO-101 assets")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument("--output", type=str, default="", help="Optional checkpoint path (pt)")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
