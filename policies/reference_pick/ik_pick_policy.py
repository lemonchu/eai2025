from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import sapien


@dataclass
class JointStage:
    name: str
    joint_targets: np.ndarray
    gripper_target: float


def _quat_from_yaw(yaw: float) -> np.ndarray:
    half = yaw / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny, cosy))


class IKPickPolicy:
    """Reference-inspired pick policy that computes IK waypoints per episode."""

    def __init__(
        self,
        env,
        arm_name: str = "right",
        approach_height: float = 0.18,
        descend_offset: float = 0.01,
        lift_height: float = 0.20,
        tolerance: float = 0.03,
        gripper_open: float = 0.6,
        gripper_close: float = -0.1,
    ) -> None:
        self.env = env
        self.arm_name = arm_name
        self.approach_height = approach_height
        self.descend_offset = descend_offset
        self.lift_height = lift_height
        self.tolerance = tolerance
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self._stages: List[JointStage] = []
        self._stage_idx = 0
        self._cached_zero: np.ndarray | None = None

        unwrapped = env.unwrapped
        self._arms = list(getattr(unwrapped, "_arms", []))
        if not self._arms:
            raise RuntimeError("Environment does not expose ArmInterface objects")
        lookup = {arm.name: idx for idx, arm in enumerate(self._arms)}
        if arm_name not in lookup:
            raise ValueError(f"Arm '{arm_name}' not found. Available: {list(lookup)}")
        self.arm_index = lookup[arm_name]
        self.arm = self._arms[self.arm_index]
        self.dof_per_arm = len(getattr(unwrapped, "_controlled_joint_names"))
        self.action_scale = getattr(unwrapped, "_action_scale", 0.05)
        self.total_dofs = env.action_space.shape[0]

    @classmethod
    def from_env(cls, env, **kwargs) -> "IKPickPolicy":
        return cls(env, **kwargs)

    def reset(self) -> None:
        self._stage_idx = 0
        self._plan_stages()

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
        state = obs.get("state") or {}
        arm_key = f"{self.arm_name}_joint_pos"
        if arm_key not in state:
            raise KeyError(f"Observation missing '{arm_key}'")
        joints = np.asarray(state[arm_key], dtype=np.float32)
        stage = self._stages[self._stage_idx]
        delta = np.clip(stage.joint_targets - joints, -self.action_scale, self.action_scale)
        start = self.arm_index * self.dof_per_arm
        stop = start + self.dof_per_arm
        action[start:stop] = delta
        # Override gripper entry explicitly
        action[stop - 1] = np.clip(stage.gripper_target - joints[-1], -self.action_scale, self.action_scale)
        if np.linalg.norm(joints - stage.joint_targets) < self.tolerance:
            self._stage_idx += 1
        return action

    def _plan_stages(self) -> None:
        self._stages.clear()
        unwrapped = self.env.unwrapped
        if not getattr(unwrapped, "_task_objects", None):
            return
        cube = unwrapped._task_objects[0]
        pose = cube.get_pose()
        cube_pos = np.array(pose.p, dtype=np.float32)
        cube_quat = np.array(pose.q if hasattr(pose, "q") else pose.r, dtype=np.float32)
        yaw = _yaw_from_quat(cube_quat)
        # Look straight down, rotate closing axis with cube yaw
        downward = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        orientation = _quat_multiply(_quat_from_yaw(yaw), downward)

        base_qpos = self.arm.robot.get_qpos().copy()
        current = base_qpos[self.arm.joint_indices]

        approach_pos = cube_pos.copy()
        approach_pos[2] = cube_pos[2] + self.approach_height
        descend_pos = cube_pos.copy()
        descend_pos[2] = max(cube_pos[2] + self.descend_offset, cube_pos[2] + 0.005)
        lift_pos = cube_pos.copy()
        lift_pos[2] = cube_pos[2] + self.lift_height

        stages = [
            ("approach", approach_pos, self.gripper_open),
            ("descend", descend_pos, self.gripper_open),
            ("grip", descend_pos, self.gripper_close),
            ("lift", lift_pos, self.gripper_close),
        ]

        last_full = base_qpos
        for name, target_pos, grip in stages:
            target_pose = sapien.Pose(target_pos, orientation.tolist())
            joint_targets = self._solve_ik(target_pose, last_full) or current
            joint_targets = joint_targets.astype(np.float32)
            joint_targets[-1] = grip
            stage = JointStage(name=name, joint_targets=joint_targets, gripper_target=grip)
            self._stages.append(stage)
            last_full = last_full.copy()
            last_full[self.arm.joint_indices] = joint_targets

    def _solve_ik(self, pose: sapien.Pose, initial_qpos: np.ndarray | None) -> np.ndarray | None:
        robot = self.arm.robot
        if initial_qpos is not None:
            initial = np.asarray(initial_qpos, dtype=np.float32)
        else:
            initial = None
        try:
            if initial_qpos is not None:
                qpos = robot.compute_inverse_kinematics(self.arm.eef_link, pose, initial_qpos=initial)
            else:
                qpos = robot.compute_inverse_kinematics(self.arm.eef_link, pose)
        except TypeError:
            qpos = robot.compute_inverse_kinematics(self.arm.eef_link, pose)
        except Exception:
            return None
        if qpos is None:
            return None
        if isinstance(qpos, tuple) and len(qpos) == 2:
            _, qpos = qpos
        full = np.array(qpos, dtype=np.float32)
        return full[self.arm.joint_indices]