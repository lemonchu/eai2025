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
        approach_height: float = 0.00,
        descend_offset: float = 0.01,
        lift_height: float = 0.20,
        tolerance: float = 0.03,
        gripper_open: float = 0.6,
        gripper_close: float = -0.1,
        grip_close_threshold: float = 0.035,
        grip_close_xy: float = 0.1, # Placeholder, set to 0.1 to ensure robot arm is moving
        grip_close_z: float = 0.1,
    ) -> None:
        self.env = env
        self.arm_name = arm_name
        self.approach_height = approach_height
        self.descend_offset = descend_offset
        self.lift_height = lift_height
        self.tolerance = tolerance
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.grip_close_threshold = grip_close_threshold
        self.grip_close_xy = grip_close_xy
        self.grip_close_z = grip_close_z
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
        # Decide gripper command: only close when the EEF is near the cube
        desired_gripper = stage.gripper_target
        if stage.name == "grip":
            # get current cube position
            cube_pos = None
            try:
                if hasattr(self, "_current_cube_pos") and self._current_cube_pos is not None:
                    cube_pos = self._current_cube_pos
                else:
                    unwrapped = self.env.unwrapped
                    cube = getattr(unwrapped, "_task_objects", [None])[0]
                    if cube is not None:
                        cube_pos = np.array(cube.get_pose().p, dtype=np.float32)
            except Exception:
                cube_pos = None
            try:
                eef_pos = np.array(self.arm.eef_link.get_pose().p)
            except Exception:
                eef_pos = None
            if cube_pos is None or eef_pos is None:
                # fallback to radial distance if we can't get both positions
                try:
                    if eef_pos is not None and cube_pos is not None:
                        dist = float(np.linalg.norm(eef_pos - cube_pos))
                    else:
                        dist = float("inf")
                except Exception:
                    dist = float("inf")
                if dist > float(self.grip_close_threshold):
                    desired_gripper = self.gripper_open
                    print(f"[GRIP-GATE] stage=grip eef_dist={dist:.4f} > {self.grip_close_threshold:.3f}; keeping gripper open")
                    print(
                        f"[GRIP-DIAG] eef={None if eef_pos is None else eef_pos.tolist()} cube={None if cube_pos is None else cube_pos.tolist()} "
                        f"gripper_current={joints[-1]:.4f} planned_close={stage.gripper_target:.4f} desired={desired_gripper:.4f}"
                    )
                else:
                    print(f"[GRIP-GATE] stage=grip eef_dist={dist:.4f} <= {self.grip_close_threshold:.3f}; allowing close")
                    print(
                        f"[GRIP-DIAG] eef={None if eef_pos is None else eef_pos.tolist()} cube={None if cube_pos is None else cube_pos.tolist()} "
                        f"gripper_current={joints[-1]:.4f} planned_close={stage.gripper_target:.4f} desired={desired_gripper:.4f}"
                    )
            else:
                # compute planar and vertical separation
                eef_xy = np.asarray(eef_pos[:2], dtype=np.float32)
                cube_xy = np.asarray(cube_pos[:2], dtype=np.float32)
                planar = float(np.linalg.norm(eef_xy - cube_xy))
                vertical = float(abs(float(eef_pos[2]) - float(cube_pos[2])))
                xy_ok = planar <= float(self.grip_close_xy)
                z_ok = vertical <= float(self.grip_close_z)
                if not (xy_ok and z_ok):
                    desired_gripper = self.gripper_open
                    print(
                        f"[GRIP-GATE] stage=grip planar={planar:.4f} (thr={self.grip_close_xy:.3f}) vertical={vertical:.4f} (thr={self.grip_close_z:.3f}) -> keeping gripper open"
                    )
                    print(
                        f"[GRIP-DIAG] eef={eef_pos.tolist()} cube={cube_pos.tolist()} "
                        f"gripper_current={joints[-1]:.4f} planned_close={stage.gripper_target:.4f} desired={desired_gripper:.4f}"
                    )
                else:
                    print(
                        f"[GRIP-GATE] stage=grip planar={planar:.4f} vertical={vertical:.4f} -> allowing close"
                    )
                    print(
                        f"[GRIP-DIAG] eef={eef_pos.tolist()} cube={cube_pos.tolist()} "
                        f"gripper_current={joints[-1]:.4f} planned_close={stage.gripper_target:.4f} desired={desired_gripper:.4f}"
                    )

        # Override gripper entry explicitly (use desired_gripper)
        gripper_action = np.clip(desired_gripper - joints[-1], -self.action_scale, self.action_scale)
        action[stop - 1] = gripper_action
        if stage.name == "grip":
            print(f"[GRIP-ACTION] value={gripper_action:.6f} desired_gripper={desired_gripper:.4f} current={joints[-1]:.4f}")

        # For progress checking, use a temporary joint-target vector that
        # reflects the desired_gripper (so gating doesn't wait for a target
        # we're intentionally not commanding yet).
        progress_targets = stage.joint_targets.copy()
        try:
            progress_targets[-1] = desired_gripper
        except Exception:
            pass

        # Advance stage only when joint-space target (including gripper) is reached
        progress_ok = np.linalg.norm(joints - progress_targets) < self.tolerance
        if stage.name == "grip" and desired_gripper != stage.gripper_target:
            # We are intentionally gating the gripper open; do not advance
            # to the next stage even if other joints match their targets.
            if progress_ok:
                print(
                    f"[GRIP-PROGRESS-BLOCKED] joints match progress targets but gripper gated open;"
                    f" eef_dist_block={self.grip_close_threshold:.4f}"
                )
        else:
            if progress_ok:
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
        default_down_orientation = _quat_multiply(_quat_from_yaw(yaw), downward)
        # capture current EEF orientation so the approach stage can keep it
        try:
            current_eef_pose = self.arm.eef_link.get_pose()
            current_eef_q = np.array(current_eef_pose.q, dtype=np.float32)
        except Exception:
            current_eef_q = default_down_orientation

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
        q_yaw = _quat_from_yaw(yaw)
        for name, target_pos, grip in stages:
            # choose per-stage orientation:
            # - approach: yaw-only (align yaw with cube, allow free pitch/roll)
            # - descend/grip: downward-facing orientation for reliable grasp
            # - others: keep current EEF orientation
            if name == "approach":
                orientation_stage = q_yaw
            elif name in ("descend", "grip"):
                orientation_stage = default_down_orientation
            else:
                orientation_stage = current_eef_q
            target_pose = sapien.Pose(target_pos, orientation_stage.tolist())
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