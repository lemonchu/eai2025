from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import sapien
from PIL import Image
from gymnasium import spaces
from sapien.render import RenderCameraComponent

from .layout import TableLayout, spawn_layout
from .tasks import PickCubeTask, PickCubeTaskConfig


# CameraSpec: width/height/fovy
@dataclass
class CameraSpec:
    width: int = 640
    height: int = 480
    fovx_deg: float = 117.12
    fovy_deg: float = 73.63
    near: float = 0.01
    far: float = 5.0

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def fovx_rad(self) -> float:
        return np.deg2rad(self.fovx_deg)

    @property
    def fovy_rad(self) -> float:
        # Derive the vertical FOV from the horizontal one to avoid stretched renders.
        return 2.0 * np.arctan(np.tan(self.fovx_rad / 2.0) / self.aspect_ratio)
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
        task: str | None = None,
    ) -> None:
        super().__init__()
        self._render_mode = render_mode
        self._headless = headless
        if scene_root is not None:
            self._scene_root = Path(scene_root)
        else:
            assets_root = Path(__file__).resolve().parents[2] / "scene" / "assets"
            candidate = assets_root / "SO101"
            if not candidate.exists():
                raise FileNotFoundError(
                    "Unable to locate SO-101 assets. Provide scene_root or place the assets "
                    "under scene/assets/SO101."
                )
            self._scene_root = candidate
        self._max_episode_steps = max_episode_steps
        self._action_scale = action_scale
        self._success_radius = success_radius
        self._rng = np.random.default_rng()
        self._camera_spec = CameraSpec()
        self._layout = TableLayout()
        self._layout_actors: dict[str, list[Any]] | None = None
        # Task controls which objects are spawned on reset: 'lift', 'stack', 'sort' or None
        self._task = task
        self._task_objects: list[sapien.Actor] = []
        self._object_half = 0.015
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
        self._pick_cube_task: PickCubeTask | None = None
        self._last_task_info: dict[str, Any] = {}
        self._setup_scene()
        self._setup_robots()
        self._setup_cameras()
        if (self._task or "").lower() == "pick_cube":
            self._pick_cube_task = PickCubeTask(
                scene=self.scene,
                layout=self._layout,
                rng=self._rng,
                config=PickCubeTaskConfig(),
            )
        self._setup_objects()
        self._define_spaces()
        self._last_distance = 1e6
        self._step_count = 0

    def _setup_scene(self) -> None:
        self.scene = sapien.Scene()
        self.scene.set_timestep(1 / 240)
        # Ground material: use a soft, slightly warm/gray base instead of pure white
        try:
            ground_material = sapien.render.RenderMaterial()
            # Try to load ground albedo and roughness maps from the scene assets directory.
            # Fixed paths under the scene assets directory: ./textures
            # Get current folder   
            current_folder = Path(__file__).parent
            albedo_path =  current_folder / "textures/ground_texture2.jpg"
            roughness_path = current_folder / "textures/ground_texture.jpg"
            print("Loading ground textures from:", albedo_path, roughness_path)

            # Fallback defaults
            base_color = np.array([0.95, 0.95, 0.96, 1.0], dtype=np.float32)
            roughness_val = 0.9

            if albedo_path.exists():
                try:
                    img = Image.open(albedo_path).convert("RGB")
                    arr = np.asarray(img).astype(np.float32) / 255.0
                    avg = arr.mean(axis=(0, 1))
                    base_color = np.array([avg[0], avg[1], avg[2], 1.0], dtype=np.float32)
                    print('Loaded ground albedo average color:', base_color)
                except Exception:
                    pass

            if roughness_path.exists():
                try:
                    rimg = Image.open(roughness_path).convert("L")
                    rarr = np.asarray(rimg).astype(np.float32) / 255.0
                    roughness_val = float(rarr.mean())
                    print('Loaded ground roughness average value:', roughness_val)
                except Exception:
                    pass

            ground_material.base_color = base_color
            # apply roughness/specular
            try:
                ground_material.roughness = roughness_val
            except Exception:
                # some Sapien builds may not expose roughness attribute
                pass
            try:
                ground_material.specular = 0.05
            except Exception:
                pass
            self.scene.add_ground(-1, render_material=ground_material)
        except Exception as e:
            # Fallback to default ground if RenderMaterial isn't available in this build
            print(e)
            self.scene.add_ground(-1)

        # Softer ambient light with a slight cool tint to avoid a pure-white background
        self.scene.set_ambient_light([0.22, 0.22, 0.24])

        # Key light: moderate intensity, slightly warm, produces softer highlights
        self.scene.add_directional_light([0.2, -1.0, -1.0], color=[1.2, 1.05, 0.95], shadow=True)

        # Fill light: soft cool fill to remove harsh white shadows
        self.scene.add_directional_light([-0.5, 0.8, -0.6], color=[0.5, 0.55, 0.65], shadow=False)
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

        class _SimpleAgent:
            def __init__(self, uid, robot):
                self.uid = uid
                self.robot = robot

            @property
            def active_joints(self):
                return self.robot.get_active_joints()

        # Simple agent placeholder for compatibility
        if hasattr(self, "robot") and self.robot is not None:
            self.agent = _SimpleAgent(uid="so101", robot=self.robot)

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

        links = {link.name: link for link in robot.get_links()}
        eef_candidates = (
            "gripper_frame_link",  # legacy SO101 URDF
            "gripper_link_tip",  # helper link exposed by the new SO101 URDF
            "moving_jaw_so101_v1_link_tip",
            "gripper_link",  # final fallback so the env at least loads
        )
        eef_link = next((links[name] for name in eef_candidates if name in links), None)
        if eef_link is None:
            available = ", ".join(sorted(links.keys()))
            raise RuntimeError(
                "Failed to locate an end-effector link in URDF. "
                f"Looked for {eef_candidates}, available links: {available}"
            )

        camera_link = next((links[name] for name in links if "camera_link" in name), None)
        if camera_link is None:
            raise RuntimeError("SO-101 URDF is missing a camera_link for the wrist camera mount")
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
        camera.set_fovx(spec.fovx_rad, compute_y=False)
        camera.set_fovy(spec.fovy_rad, compute_x=False)
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

        # Build sensors mapping for compatibility
        self._sensors = {}
        # front: RenderCameraComponent -> wrap to present same methods as mounted_camera
        self._sensors["front"] = self._front_camera

        # Add top/side cameras using layout helpers (requires the layout helper methods from layout.py)
        # If you added layout.top_camera_pose/side_camera_pose, use them; else craft poses similar to maniskill_adapter
        try:
            top_pose = self._layout.top_camera_pose()
            side_pose = self._layout.side_camera_pose()
            self._top_camera = self.scene.add_mounted_camera(
                name="top_camera",
                mount=mount,
                pose=top_pose,
                width=spec.width,
                height=spec.height,
                fovy=np.deg2rad(73.63),
                near=spec.near,
                far=spec.far,
            )
            self._side_camera = self.scene.add_mounted_camera(
                name="side_camera",
                mount=mount,
                pose=side_pose,
                width=spec.width,
                height=spec.height,
                fovy=np.deg2rad(73.63),
                near=spec.near,
                far=spec.far,
            )
            # Also add an additional right-side camera positioned a bit further right
            try:
                # copy pose and offset slightly in +X (world) to move further to the right
                side_pose_right = self._layout.side_camera_pose()
                try:
                    pos = np.array(side_pose_right.p)
                    quat = np.array(side_pose_right.q) if hasattr(side_pose_right, "q") else None
                except Exception as e:
                    pos = np.zeros(3)
                    quat = None
                pos_right = pos

                # Apply a small downward pitch to the right-side camera so it looks
                # roughly horizontal or slightly downward onto the table.
                # We'll create a pitch quaternion (rotation about local X) and
                # multiply it with the existing pose quaternion (if present).
                def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                    # a and b are [w, x, y, z]
                    aw, ax, ay, az = a
                    bw, bx, by, bz = b
                    return np.array([
                        aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                    ], dtype=float)

                pitch = 0
                half = float(np.cos(pitch / 2.0))
                sin_half = float(np.sin(pitch / 2.0))
                q_pitch = np.array([half, 0.0, 0.0, sin_half], dtype=float)
                if quat is None:
                    q_world = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                else:
                    q_world = quat.astype(float)
                q_new = _quat_mul(q_world, q_pitch)
                right_pose = sapien.Pose(pos_right, q_new)
                self._right_side_camera = self.scene.add_mounted_camera(
                    name="right_side_camera",
                    mount=mount,
                    pose=right_pose,
                    width=spec.width,
                    height=spec.height,
                    fovy=np.deg2rad(73.63),
                    near=spec.near,
                    far=spec.far,
                )
                self._sensors["right_side"] = self._right_side_camera
            except Exception:
                # If creating the extra camera fails, continue without it
                pass
            self._sensors["top"] = self._top_camera
            self._sensors["side"] = self._side_camera
        except Exception:
            # Fallback: keep only front + wrist cameras
            pass

        # Wrist cameras: we keep one mapping 'wrist' to the first arm's wrist for legacy code,
        # and also per-arm '{arm}_wrist' keys already stored in self._wrist_cameras.
        if self._wrist_cameras:
            first_arm_name = next(iter(self._wrist_cameras.keys()))
            self._sensors["wrist"] = self._wrist_cameras[first_arm_name]
        for arm in self._arms:
            key = f"{arm.name}_wrist"
            self._sensors[key] = self._wrist_cameras[arm.name]

    def _setup_objects(self) -> None:
        self._layout_actors = spawn_layout(self.scene, self._layout)
        # Task objects will be spawned at reset; keep a list for cleanup
        self._task_objects = []
        self._object_half = 0.015

    def _define_spaces(self) -> None:
        image_shape = (self._camera_spec.height, self._camera_spec.width, 3)
        image_spaces = {
            "front": spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
        }
        # include right-side camera image if present
        image_spaces["right_side"] = spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
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
        self._last_task_info = {}
        self._reset_robot_state()
        # Clear and spawn task-specific objects (if any)
        self._clear_task_objects()
        self._spawn_task_objects()
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
        # Deprecated: individual cube placement is handled by _spawn_task_objects
        return

    def _clear_task_objects(self) -> None:
        if self._pick_cube_task is not None:
            self._pick_cube_task.clear()
        for a in list(getattr(self, "_task_objects", []) or []):
            try:
                if hasattr(a, "release"):
                    a.release()
                elif hasattr(self.scene, "remove_actor"):
                    # best-effort removal
                    self.scene.remove_actor(a)
            except Exception:
                # ignore failures during cleanup
                pass
        self._task_objects = []

    def _spawn_task_objects(self) -> None:
        """Spawn objects depending on `self._task`:
        - 'lift': one red cube in the right robot area
        - 'stack': two cubes (red bottom, green top) stacked in the right robot area
        - 'sort': two cubes (red, green) in the center bin area
        """
        if self._pick_cube_task is not None:
            self._task_objects = list(self._pick_cube_task.spawn())
            return
        task = (self._task or "lift").lower()
        half = self._object_half
        # helper to build a colored cube (static) with an optional yaw rotation around Z
        def build_cube(color, position, name, yaw: float | None = None):
            b = self.scene.create_actor_builder()
            b.add_box_visual(half_size=[half, half, half], material=list(color))
            b.add_box_collision(half_size=[half, half, half])
            actor = b.build_static(name=name)
            # random yaw if not provided
            if yaw is None:
                yaw = float(self._rng.uniform(-np.pi, np.pi))
            qw = float(np.cos(yaw / 2.0))
            qz = float(np.sin(yaw / 2.0))
            quat = [qw, 0.0, 0.0, qz]
            actor.set_pose(sapien.Pose(position, quat))
            return actor

        if task == "stack":
            # Spawn two separate cubes in the right robot area (robot must stack them)
            pos1 = self._layout.sample_pick_region(self._rng, bin_index=2, area="bin")
            pos2 = self._layout.sample_pick_region(self._rng, bin_index=2, area="bin")
            # ensure they are not overlapping
            if np.linalg.norm(pos1[:2] - pos2[:2]) < 0.06:
                pos2[:2] += np.array([0.06, 0.0])
            pos1[2] = self._layout.table_surface_z + half
            pos2[2] = self._layout.table_surface_z + half
            a1 = build_cube([1, 0, 0], pos1, "cube_red")
            a2 = build_cube([0, 1, 0], pos2, "cube_green")
            self._task_objects.extend([a1, a2])
            return

        if task == "sort":
            # Two cubes (red and green) in center bin with random yaw
            pos1 = self._layout.sample_pick_region(self._rng, bin_index=1, area="bin")
            pos2 = self._layout.sample_pick_region(self._rng, bin_index=1, area="bin")
            # ensure not too close
            if np.linalg.norm(pos1[:2] - pos2[:2]) < 0.03:
                pos2[:2] += np.array([0.03, 0.03])
            pos1[2] = self._layout.table_surface_z + half
            pos2[2] = self._layout.table_surface_z + half
            a1 = build_cube([1, 0, 0], pos1, "cube_red", yaw=None)
            a2 = build_cube([0, 1, 0], pos2, "cube_green", yaw=None)
            self._task_objects.extend([a1, a2])
            return

        # Default / 'lift': single red cube in the right robot area with random yaw
        sample = self._layout.sample_pick_region(self._rng, bin_index=2, area="bin")
        sample[2] = self._layout.table_surface_z + half
        a = build_cube([1, 0, 0], sample, "cube_red", yaw=None)
        self._task_objects.append(a)

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
        # Success criterion: object lifted by at least configured threshold.
        # Prefer explicit physics/pose-based check rather than proximity.
        success = self.check_success()
        self._step_count += 1
        terminated = success
        truncated = self._step_count >= self._max_episode_steps
        # Report object lift heights for easier debugging/metrics
        max_lift = float(0.0)
        lifts = []
        for a in getattr(self, "_task_objects", []):
            try:
                p = np.array(a.get_pose().p)
                lift_h = float(p[2] - self._object_half - self._layout.table_surface_z)
                lifts.append(lift_h)
            except Exception:
                continue
        if lifts:
            max_lift = float(max(lifts))
        # Include gripped flag (set by _compute_reward)
        gripped = bool(getattr(self, "_last_gripped", False))
        info = {"success": success, "distance": self._last_distance, "max_lift": max_lift, "gripped": gripped}
        if self._last_task_info:
            info.update(self._last_task_info)
        if self._viewer is not None:
            self._viewer.render()
        return obs, reward, terminated, truncated, info

    def _compute_reward(self) -> float:
        # Reward based on object lift above the table surface (meters), plus a grasp bonus.
        # We still compute and update `self._last_distance` for compatibility/debug info.
        if self._pick_cube_task is not None:
            reward, metrics = self._pick_cube_task.compute_reward(self._arms)
            self._last_task_info = metrics.copy()
            if "distance" in metrics:
                self._last_distance = float(metrics["distance"])
            self._last_gripped = bool(metrics.get("is_grasped", False))
            return reward

        self._last_task_info = {}
        max_lift = 0.0
        object_positions: list[np.ndarray] = []
        for a in getattr(self, "_task_objects", []):
            try:
                p = np.array(a.get_pose().p)
                lift_h = float(p[2] - self._object_half - self._layout.table_surface_z)
                object_positions.append(p)
                if lift_h > max_lift:
                    max_lift = lift_h
            except Exception:
                pass

        # Update last_distance for compatibility/logging (min distance from any eef to any object)
        distances: list[float] = []
        for arm in self._arms:
            try:
                eef_pos = np.array(arm.eef_link.get_pose().p)
                for obj_pos in object_positions:
                    distances.append(float(np.linalg.norm(eef_pos - obj_pos)))
            except Exception:
                continue
        if distances:
            self._last_distance = float(min(distances))
        else:
            self._last_distance = 1e6

        # Detect a simple approximate grasp: gripper joint closed AND eef close to an object.
        gripped = False
        grasp_distance_thresh = 0.02  # meters
        gripper_closed_thresh = 0.0   # values <= this are considered 'closed'
        for arm in self._arms:
            try:
                qpos = arm.robot.get_qpos()
                gripper_val = float(qpos[arm.joint_indices[-1]])
                if gripper_val <= gripper_closed_thresh:
                    eef_pos = np.array(arm.eef_link.get_pose().p)
                    for obj_pos in object_positions:
                        if float(np.linalg.norm(eef_pos - obj_pos)) <= grasp_distance_thresh:
                            gripped = True
                            break
                if gripped:
                    break
            except Exception:
                continue

        # Store last gripped state for info reporting
        self._last_gripped = bool(gripped)

        # Shaped reward: proportional to lift (meters), non-negative, plus bonuses
        reward = float(max(0.0, max_lift))
        if max_lift >= 0.05:
            reward += 1.0
        if gripped:
            # Encourage establishing a grasp (smaller bonus than successful lift)
            reward += 0.5
        return reward

    def _get_observation(self) -> dict[str, Any]:
        front, wrist_frames, right = self._render_cameras()
        images: dict[str, np.ndarray] = {"front": front}
        # include right-side view
        images["right_side"] = right.copy() if isinstance(right, np.ndarray) else right
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
        # Right-side camera (mounted) if available
        right_rgb = None
        try:
            if hasattr(self, "_right_side_camera") and self._right_side_camera is not None:
                # mounted_camera object
                self._right_side_camera.take_picture()
                rcolor = self._right_side_camera.get_picture("Color")
                # convert to uint8
                right_rgb = self._float_to_uint8(rcolor)
        except Exception:
            right_rgb = None

        if right_rgb is None:
            # fallback to an empty image with same shape as front
            try:
                right_rgb = np.zeros_like(self._front_rgb)
            except Exception:
                right_rgb = np.zeros((self._camera_spec.height, self._camera_spec.width, 3), dtype=np.uint8)

        return self._front_rgb.copy(), wrist_frames, right_rgb

    @staticmethod
    def _float_to_uint8(rgba: np.ndarray) -> np.ndarray:
        rgb = np.clip(rgba[..., :3], 0.0, 1.0) * 255.0
        return rgb.astype(np.uint8)

    def render(self) -> np.ndarray:
        if self._render_mode != "rgb_array":
            raise NotImplementedError("Only rgb_array render mode is supported in headless mode")
        self.scene.update_render()
        frame, _, _ = self._render_cameras()
        return frame

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self.scene = None
        self.robots = []
        self.robot = None
        self._arms = []

    def take_picture(self, camera_name: str):
        """Compatibility method: returns an RGBA uint8 image for a named camera."""
        self.scene.update_render()
        if camera_name not in self._sensors:
            raise KeyError(f"Camera '{camera_name}' not found. Available: {list(self._sensors.keys())}")
        cam = self._sensors[camera_name]
        # Some cam objects have take_picture/get_picture, others use RenderCameraComponent API
        try:
            cam.take_picture()
            color = cam.get_picture("Color")
        except Exception:
            # RenderCameraComponent path (front camera implementation)
            if hasattr(cam, "take_picture"):
                cam.take_picture()
                color = cam.get_picture("Color")
            else:
                raise RuntimeError("Unsupported camera object type for take_picture")
        # color as float [H,W,3] in [0,1] -> convert to uint8 RGB
        rgb = np.clip(color[..., :3], 0.0, 1.0) * 255.0
        rgb_uint8 = rgb.astype(np.uint8)
        h, w, _ = rgb_uint8.shape
        rgba = np.full((h, w, 4), 255, dtype=np.uint8)
        rgba[:, :, :3] = rgb_uint8
        return rgba

    def get_obs_compat(self):
        """
        Return a compatibility dict similar to ManiSkill example:
        {'qpos': <ndarray>, 'cube_pos': <ndarray>, 'cube_quat': <ndarray>}
        """
        # qpos: use first arm's active joints
        if not self._arms:
            raise RuntimeError("No arms available")
        arm = self._arms[0]
        qpos_full = arm.robot.get_qpos()
        qpos = qpos_full[arm.joint_indices].astype(np.float32)
        # cube pose
        # Prefer the first task object (if any) for compatibility
        cube_pos = np.zeros(3, dtype=np.float32)
        cube_quat = np.array([1, 0, 0, 0], dtype=np.float32)
        if getattr(self, "_task_objects", None):
            try:
                p = self._task_objects[0].get_pose()
                cube_pos = np.array(p.p, dtype=np.float32)
                cube_quat = np.array(p.r, dtype=np.float32) if hasattr(p, "r") else np.array(p.q, dtype=np.float32)
            except Exception:
                pass
        # Normalize shapes (flatten)
        cube_pos = cube_pos.flatten()
        cube_quat = cube_quat.flatten()
        return {"qpos": qpos, "cube_pos": cube_pos, "cube_quat": cube_quat}

    def check_success(self) -> bool:
        if self._pick_cube_task is not None:
            return self._pick_cube_task.check_success()
        for a in getattr(self, "_task_objects", []):
            try:
                p = np.array(a.get_pose().p, dtype=np.float32).flatten()
                lift_height = p[2] - self._object_half
                if float(lift_height) >= 0.05:
                    return True
            except Exception:
                continue
        return False