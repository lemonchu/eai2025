from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import sapien
from gymnasium import spaces
from sapien.pysapien.render import RenderCameraComponent

from .layout import TableLayout, spawn_layout


@dataclass
class CameraSpec:
    width: int = 640
    height: int = 480
    fovx_deg: float = 117.12
    fovy_deg: float = 73.63
    near: float = 0.01
    far: float = 5.0


class So101SceneEnv(gym.Env):
    """Minimal Sapien-based SO-101 scene compatible with LeRobot policies."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str = "rgb_array",
        scene_root: str | None = None,
        headless: bool = True,
        max_episode_steps: int = 200,
        action_scale: float = 0.05,
        success_radius: float = 0.035,
    ) -> None:
        super().__init__()
        self._render_mode = render_mode
        self._headless = headless
        self._scene_root = (
            Path(scene_root)
            if scene_root is not None
            else Path(__file__).resolve().parents[2] / "scene" / "assets" / "SO101"
        )
        self._max_episode_steps = max_episode_steps
        self._action_scale = action_scale
        self._success_radius = success_radius
        self._rng = np.random.default_rng()
        self._camera_spec = CameraSpec()
        self._layout = TableLayout()
        self._layout_actors: dict[str, list[Any]] | None = None
        self._setup_scene()
        self._setup_robot()
        self._setup_cameras()
        self._setup_objects()
        self._define_spaces()
        self._last_distance = 1e6
        self._step_count = 0

    def _setup_scene(self) -> None:
        self.scene = sapien.Scene()
        self.scene.set_timestep(1 / 240)
        self.scene.add_ground(0)
        self.scene.set_ambient_light([0.4, 0.4, 0.4])
        self.scene.add_directional_light([0.3, -1, -1], [2.5, 2.5, 2.5], shadow=True)
        self.scene.add_directional_light([-0.5, 1, -1], [1.5, 1.5, 1.5], shadow=False)
        self._viewer = None
        if self._render_mode == "human" and not self._headless:
            self._viewer = self.scene.create_viewer()
            self._viewer.set_camera_xyz(x=-2, y=0, z=1)
            self._viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    def _setup_robot(self) -> None:
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(str(self._scene_root / "so101.urdf"))
        if self.robot is None:
            raise FileNotFoundError(f"Failed to load URDF at {self._scene_root}")
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        self._controlled_joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        active_joints = self.robot.get_active_joints()
        joints = {joint.name: joint for joint in active_joints}
        missing = [name for name in self._controlled_joint_names if name not in joints]
        if missing:
            raise RuntimeError(f"Missing joints in URDF: {missing}")
        self._joint_indices = [active_joints.index(joints[name]) for name in self._controlled_joint_names]
        limits = [joints[name].get_limits()[0] for name in self._controlled_joint_names]
        self._joint_lower = np.array([limit[0] for limit in limits], dtype=np.float32)
        self._joint_upper = np.array([limit[1] for limit in limits], dtype=np.float32)
        self._initial_qpos = np.array([0.0, -0.4, 0.7, 0.4, 0.0, 0.4], dtype=np.float32)
        self._eef_link = next(link for link in self.robot.get_links() if link.name == "gripper_frame_link")

    def _setup_cameras(self) -> None:
        spec = self._camera_spec
        mount = sapien.Entity()
        camera = RenderCameraComponent(spec.width, spec.height)
        camera.set_fovx(np.deg2rad(spec.fovx_deg), compute_y=False)
        camera.set_fovy(np.deg2rad(spec.fovy_deg), compute_x=False)
        camera.near = spec.near
        camera.far = spec.far
        mount.add_component(camera)
        mount.name = "front_camera_mount"
        camera.name = "front_camera"
        self.scene.add_entity(mount)
        mount.set_pose(self._layout.camera_pose())
        camera.set_local_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        self._front_camera = camera
        camera_link = next(link for link in self.robot.get_links() if "camera_link" in link.name)
        self._wrist_camera = self.scene.add_mounted_camera(
            name="wrist_camera",
            mount=camera_link.entity,
            pose=sapien.Pose(np.eye(4)),
            width=spec.width,
            height=spec.height,
            fovy=np.deg2rad(70.0),
            near=spec.near,
            far=spec.far,
        )
        self._front_rgb = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
        self._wrist_rgb = np.zeros_like(self._front_rgb)

    def _setup_objects(self) -> None:
        self._layout_actors = spawn_layout(self.scene, self._layout)
        cube_builder = self.scene.create_actor_builder()
        cube_builder.add_box_visual(half_size=[0.015, 0.015, 0.015], material=[1, 0, 0])
        cube_builder.add_box_collision(half_size=[0.015, 0.015, 0.015])
        self._cube = cube_builder.build_static(name="target_cube")
        self._cube_height = 0.015

    def _define_spaces(self) -> None:
        image_shape = (self._camera_spec.height, self._camera_spec.width, 3)
        self.observation_space = spaces.Dict(
            {
                "images": spaces.Dict(
                    {
                        "front": spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
                        "wrist": spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
                    }
                ),
                "state": spaces.Dict(
                    {
                        "joint_pos": spaces.Box(self._joint_lower, self._joint_upper, dtype=np.float32),
                        "joint_vel": spaces.Box(
                            low=-np.full_like(self._joint_lower, np.inf, dtype=np.float32),
                            high=np.full_like(self._joint_upper, np.inf, dtype=np.float32),
                            dtype=np.float32,
                        ),
                    }
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-self._action_scale,
            high=self._action_scale,
            shape=(len(self._controlled_joint_names),),
            dtype=np.float32,
        )

    def seed(self, seed: int | None = None) -> list[int]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return [seed] if seed is not None else []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.seed(seed)
        self._step_count = 0
        self._reset_robot_state()
        self._reset_cube_pose()
        self.scene.step()
        self.scene.update_render()
        obs = self._get_observation()
        return obs, {"success": False}

    def _reset_robot_state(self) -> None:
        qpos = self.robot.get_qpos()
        for i, idx in enumerate(self._joint_indices):
            qpos[idx] = float(np.clip(self._initial_qpos[i], self._joint_lower[i], self._joint_upper[i]))
        self.robot.set_qpos(qpos)
        self.robot.set_qvel(np.zeros_like(qpos))

    def _reset_cube_pose(self) -> None:
        sample = self._layout.sample_pick_region(self._rng)
        sample[2] = self._layout.table_surface_z + self._cube_height
        self._cube.set_pose(sapien.Pose(sample, [1, 0, 0, 0]))

    def step(self, action: np.ndarray):
        clipped = np.clip(action, self.action_space.low, self.action_space.high)
        qpos = self.robot.get_qpos()
        for i, idx in enumerate(self._joint_indices):
            target = np.clip(qpos[idx] + clipped[i], self._joint_lower[i], self._joint_upper[i])
            qpos[idx] = float(target)
        self.robot.set_qpos(qpos)
        self.robot.set_qvel(np.zeros_like(qpos))
        for _ in range(4):
            self.scene.step()
        self.scene.update_render()
        obs = self._get_observation()
        reward = self._compute_reward()
        success = self._last_distance < self._success_radius
        self._step_count += 1
        terminated = success
        truncated = self._step_count >= self._max_episode_steps
        info = {"success": success, "distance": self._last_distance}
        if self._viewer is not None:
            self._viewer.render()
        return obs, reward, terminated, truncated, info

    def _compute_reward(self) -> float:
        eef_pos = self._eef_link.get_pose().p
        cube_pos = self._cube.get_pose().p
        self._last_distance = float(np.linalg.norm(np.array(eef_pos) - np.array(cube_pos)))
        shaped = -self._last_distance
        if self._last_distance < self._success_radius:
            shaped += 1.0
        return shaped

    def _get_observation(self) -> dict[str, Any]:
        front, wrist = self._render_cameras()
        joint_pos = self.robot.get_qpos()[self._joint_indices].astype(np.float32)
        joint_vel = self.robot.get_qvel()[self._joint_indices].astype(np.float32)
        return {
            "images": {"front": front, "wrist": wrist},
            "state": {"joint_pos": joint_pos, "joint_vel": joint_vel},
        }

    def _render_cameras(self) -> tuple[np.ndarray, np.ndarray]:
        if self._front_camera is not None:
            self._front_camera.take_picture()
            color = self._front_camera.get_picture("Color")
            self._front_rgb = self._float_to_uint8(color)
        if self._wrist_camera is not None:
            self._wrist_camera.take_picture()
            color = self._wrist_camera.get_picture("Color")
            self._wrist_rgb = self._float_to_uint8(color)
        return self._front_rgb.copy(), self._wrist_rgb.copy()

    @staticmethod
    def _float_to_uint8(rgba: np.ndarray) -> np.ndarray:
        rgb = np.clip(rgba[..., :3], 0.0, 1.0) * 255.0
        return rgb.astype(np.uint8)

    def render(self) -> np.ndarray:
        if self._render_mode != "rgb_array":
            raise NotImplementedError("Only rgb_array render mode is supported in headless mode")
        self.scene.update_render()
        frame, _ = self._render_cameras()
        return frame

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self.scene = None
        self.robot = None