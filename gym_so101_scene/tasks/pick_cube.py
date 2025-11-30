from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import sapien

from ..layout import TableLayout


@dataclass
class PickCubeTaskConfig:
    """Configuration values mirroring the TA reference task."""

    spawn_bin_index: int = 2  # right-most bin near the right arm
    spawn_jitter_xy: float = 0.03
    cube_half_size: float = 0.015
    goal_radius: float = 0.02
    max_goal_height: float = 0.08
    grasp_distance: float = 0.03
    static_joint_vel: float = 0.2


class PickCubeTask:
    """Self-contained spawn/reward logic for the PickCubeSO101 task."""

    def __init__(
        self,
        scene: sapien.Scene,
        layout: TableLayout,
        rng: np.random.Generator,
        config: PickCubeTaskConfig | None = None,
    ) -> None:
        self.scene = scene
        self.layout = layout
        self.rng = rng
        self.cfg = config or PickCubeTaskConfig()
        self.cube: Optional[sapien.Actor] = None
        self.goal_site: Optional[sapien.Actor] = None
        self._last_metrics: Dict[str, float | bool] = {}

    def clear(self) -> None:
        for actor in (self.cube, self.goal_site):
            try:
                if actor is None:
                    continue
                if hasattr(actor, "release"):
                    actor.release()
                else:
                    self.scene.remove_actor(actor)
            except Exception:
                pass
        self.cube = None
        self.goal_site = None

    def spawn(self) -> List[sapien.Actor]:
        self.clear()
        cube_pose = self._sample_cube_pose()
        self.cube = self._build_cube(cube_pose)
        self.goal_site = self._build_goal(cube_pose.p)
        self._last_metrics = {}
        return [self.cube]

    # ------------------------------------------------------------------
    # Reward / metrics
    # ------------------------------------------------------------------
    def compute_reward(self, arms: List) -> tuple[float, Dict[str, float | bool]]:
        metrics: Dict[str, float | bool] = {
            "is_obj_placed": False,
            "is_robot_static": False,
            "is_grasped": False,
            "goal_distance": np.inf,
            "distance": np.inf,
        }
        if self.cube is None or self.goal_site is None:
            return 0.0, metrics

        cube_pose = self.cube.get_pose()
        cube_pos = np.array(cube_pose.p, dtype=np.float32)
        goal_pos = np.array(self.goal_site.get_pose().p, dtype=np.float32)
        goal_dist = float(np.linalg.norm(goal_pos - cube_pos))
        metrics["goal_distance"] = goal_dist
        metrics["is_obj_placed"] = goal_dist <= self.cfg.goal_radius

        eef_dist = np.inf
        joint_vel = 0.0
        for arm in arms:
            try:
                eef_pos = np.array(arm.eef_link.get_pose().p, dtype=np.float32)
                eef_dist = min(eef_dist, float(np.linalg.norm(eef_pos - cube_pos)))
            except Exception:
                pass
            try:
                qvel = arm.robot.get_qvel()[arm.joint_indices]
                joint_vel = max(joint_vel, float(np.linalg.norm(qvel)))
            except Exception:
                pass
        metrics["distance"] = float(eef_dist if np.isfinite(eef_dist) else np.inf)
        metrics["is_grasped"] = metrics["distance"] <= self.cfg.grasp_distance
        metrics["is_robot_static"] = joint_vel < self.cfg.static_joint_vel
        metrics["success"] = bool(metrics["is_obj_placed"] and metrics["is_robot_static"])

        reaching_reward = 1.0 - np.tanh(5.0 * metrics["distance"])
        place_reward = 1.0 - np.tanh(5.0 * goal_dist)
        reward = float(reaching_reward)
        if metrics["is_grasped"]:
            reward += 1.0
            reward += float(place_reward)
        if metrics["success"]:
            reward += 1.0
        self._last_metrics = metrics
        return reward, metrics

    def check_success(self) -> bool:
        if not self._last_metrics:
            return False
        return bool(self._last_metrics.get("success", False))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_cube_pose(self) -> sapien.Pose:
        pos = self.layout.sample_pick_region(
            self.rng,
            bin_index=self.cfg.spawn_bin_index,
            area="bin",
        )
        pos = np.asarray(pos, dtype=np.float32)
        pos[2] = self.layout.table_surface_z + self.cfg.cube_half_size
        yaw = float(self.rng.uniform(-np.pi, np.pi))
        quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float32)
        return sapien.Pose(pos.tolist(), quat.tolist())

    def _build_cube(self, pose: sapien.Pose) -> sapien.Actor:
        builder = self.scene.create_actor_builder()
        half = [self.cfg.cube_half_size] * 3
        builder.add_box_visual(half_size=half, material=[1, 0, 0, 1])
        builder.add_box_collision(half_size=half)
        cube = builder.build(name="pick_cube")
        cube.set_pose(pose)
        return cube

    def _build_goal(self, cube_pos: np.ndarray) -> sapien.Actor:
        offset_xy = self.rng.uniform(-self.cfg.spawn_jitter_xy, self.cfg.spawn_jitter_xy, size=2)
        goal = cube_pos.copy()
        goal[:2] += offset_xy
        goal[2] = cube_pos[2] + self.rng.uniform(0.0, self.cfg.max_goal_height)
        pose = sapien.Pose(goal.tolist(), [1, 0, 0, 0])
        builder = self.scene.create_actor_builder()
        builder.add_sphere_visual(radius=self.cfg.goal_radius, material=[0, 1, 0, 0.6])
        goal_actor = builder.build_kinematic(name="goal_site")
        goal_actor.set_pose(pose)
        return goal_actor
