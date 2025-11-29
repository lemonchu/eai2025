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


@dataclass
class ArmInterface:
    name: str
    robot: sapien.Articulation
    joint_indices: list[int]
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    initial_qpos: np.ndarray
    eef_link: sapien.Link
    camera_link: sapien.Link


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
        self._controlled_joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        self._initial_joint_positions = np.array([0.0, -0.4, 0.7, 0.4, 0.0, 0.4], dtype=np.float32)
        self._arms: list[ArmInterface] = []
        self._setup_scene()
        self._setup_robots()
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

    def _setup_robots(self) -> None:
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        base_centers = list(self._layout.robot_base_centers())
        if len(base_centers) < 2:
            raise RuntimeError("TableLayout must provide at least two robot base positions")
        arm_labels = ("left", "right")
        self.robots: list[sapien.Articulation] = []
        for idx, label in enumerate(arm_labels):
            robot = loader.load(str(self._scene_root / "so101.urdf"))
            if robot is None:
                raise FileNotFoundError(f"Failed to load URDF at {self._scene_root}")
            base_center = base_centers[idx]
            root_pose = sapien.Pose([0.0, base_center[1], 0.0], [1, 0, 0, 0])
            robot.set_root_pose(root_pose)
            arm = self._build_arm_interface(robot, label)
            self._arms.append(arm)
            self.robots.append(robot)
        if self._arms:
            self.robot = self._arms[0].robot

    def _build_arm_interface(self, robot: sapien.Articulation, label: str) -> ArmInterface:
        active_joints = robot.get_active_joints()
        joints = {joint.name: joint for joint in active_joints}
        missing = [name for name in self._controlled_joint_names if name not in joints]
        if missing:
            raise RuntimeError(f"Missing joints in URDF: {missing}")
        indices = [active_joints.index(joints[name]) for name in self._controlled_joint_names]
        limits = [joints[name].get_limits()[0] for name in self._controlled_joint_names]
        lower = np.array([limit[0] for limit in limits], dtype=np.float32)
        upper = np.array([limit[1] for limit in limits], dtype=np.float32)
        eef_link = next(link for link in robot.get_links() if link.name == "gripper_frame_link")
        camera_link = next(link for link in robot.get_links() if "camera_link" in link.name)
        return ArmInterface(
            name=label,
            robot=robot,
            joint_indices=indices,
            joint_lower=lower,
            joint_upper=upper,
            initial_qpos=self._initial_joint_positions.copy(),
            eef_link=eef_link,
            camera_link=camera_link,
        )

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
        self._wrist_cameras: dict[str, Any] = {}
        self._front_rgb = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
        self._wrist_rgbs: dict[str, np.ndarray] = {}
        for arm in self._arms:
            wrist_camera = self.scene.add_mounted_camera(
                name=f"{arm.name}_wrist_camera",
                mount=arm.camera_link.entity,
                pose=sapien.Pose(np.eye(4)),
                width=spec.width,
                height=spec.height,
                fovy=np.deg2rad(70.0),
                near=spec.near,
                far=spec.far,
            )
            self._wrist_cameras[arm.name] = wrist_camera
            self._wrist_rgbs[arm.name] = np.zeros_like(self._front_rgb)

    def _setup_objects(self) -> None:
        self._layout_actors = spawn_layout(self.scene, self._layout)
        cube_builder = self.scene.create_actor_builder()
        cube_builder.add_box_visual(half_size=[0.015, 0.015, 0.015], material=[1, 0, 0])
        cube_builder.add_box_collision(half_size=[0.015, 0.015, 0.015])
        self._cube = cube_builder.build_static(name="target_cube")
        self._cube_height = 0.015

    def _define_spaces(self) -> None:
        image_shape = (self._camera_spec.height, self._camera_spec.width, 3)
        image_spaces = {
            "front": spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
        }
        for arm in self._arms:
            image_spaces[f"{arm.name}_wrist"] = spaces.Box(
                low=0, high=255, shape=image_shape, dtype=np.uint8
            )

        state_spaces = {}
        for arm in self._arms:
            state_spaces[f"{arm.name}_joint_pos"] = spaces.Box(
                arm.joint_lower, arm.joint_upper, dtype=np.float32
            )
            vel_limit = np.full_like(arm.joint_lower, np.inf, dtype=np.float32)
            state_spaces[f"{arm.name}_joint_vel"] = spaces.Box(
                low=-vel_limit,
                high=vel_limit,
                dtype=np.float32,
            )

        self.observation_space = spaces.Dict(
            {
                "images": spaces.Dict(image_spaces),
                "state": spaces.Dict(state_spaces),
            }
        )
        joint_dim = len(self._controlled_joint_names) * len(self._arms)
        self.action_space = spaces.Box(
            low=-self._action_scale,
            high=self._action_scale,
            shape=(joint_dim,),
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
        for arm in self._arms:
            qpos = arm.robot.get_qpos()
            for i, idx in enumerate(arm.joint_indices):
                target = np.clip(arm.initial_qpos[i], arm.joint_lower[i], arm.joint_upper[i])
                qpos[idx] = float(target)
            arm.robot.set_qpos(qpos)
            arm.robot.set_qvel(np.zeros_like(qpos))

    def _reset_cube_pose(self) -> None:
        sample = self._layout.sample_pick_region(self._rng)
        sample[2] = self._layout.table_surface_z + self._cube_height
        self._cube.set_pose(sapien.Pose(sample, [1, 0, 0, 0]))

    def step(self, action: np.ndarray):
        clipped = np.clip(action, self.action_space.low, self.action_space.high)
        per_arm = len(self._controlled_joint_names)
        splits = [clipped[i * per_arm:(i + 1) * per_arm] for i in range(len(self._arms))]
        for arm, delta in zip(self._arms, splits):
            qpos = arm.robot.get_qpos()
            for offset, idx in enumerate(arm.joint_indices):
                target = np.clip(qpos[idx] + delta[offset], arm.joint_lower[offset], arm.joint_upper[offset])
                qpos[idx] = float(target)
            arm.robot.set_qpos(qpos)
            arm.robot.set_qvel(np.zeros_like(qpos))
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
        cube_pos = self._cube.get_pose().p
        distances = []
        for arm in self._arms:
            eef_pos = arm.eef_link.get_pose().p
            distances.append(np.linalg.norm(np.array(eef_pos) - np.array(cube_pos)))
        self._last_distance = float(min(distances))
        shaped = -self._last_distance
        if self._last_distance < self._success_radius:
            shaped += 1.0
        return shaped

    def _get_observation(self) -> dict[str, Any]:
        front, wrist_frames = self._render_cameras()
        images: dict[str, np.ndarray] = {"front": front}
        state_dict: dict[str, np.ndarray] = {}
        for arm in self._arms:
            wrist_key = f"{arm.name}_wrist"
            images[wrist_key] = wrist_frames[wrist_key]
            joint_pos = arm.robot.get_qpos()[arm.joint_indices].astype(np.float32)
            joint_vel = arm.robot.get_qvel()[arm.joint_indices].astype(np.float32)
            state_dict[f"{arm.name}_joint_pos"] = joint_pos
            state_dict[f"{arm.name}_joint_vel"] = joint_vel
        return {"images": images, "state": state_dict}

    def _render_cameras(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if self._front_camera is not None:
            self._front_camera.take_picture()
            color = self._front_camera.get_picture("Color")
            self._front_rgb = self._float_to_uint8(color)
        wrist_frames: dict[str, np.ndarray] = {}
        for name, camera in self._wrist_cameras.items():
            camera.take_picture()
            color = camera.get_picture("Color")
            self._wrist_rgbs[name] = self._float_to_uint8(color)
            wrist_frames[f"{name}_wrist"] = self._wrist_rgbs[name].copy()
        return self._front_rgb.copy(), wrist_frames

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
        self.robots = []
        self.robot = None
        self._arms = []