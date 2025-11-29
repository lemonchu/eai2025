# ManiSkill Reference for SO-101 (Track 1)

This note explains how to use the official ManiSkill pick-and-place benchmark (SO-100) as a starting point for Track 1. It highlights the adjustments needed for the SO-101 arm, points to the assets that already live in this repo, and lists quick validation steps so you can confirm the setup before spending time on policy work.

## 1. Baseline: stock ManiSkill pick-and-place (SO-100)
1. Use a fresh conda env (Python 3.10 works with both LeRobot and ManiSkill 3.0+).
2. Install ManiSkill (GPU build recommended for Sapien rendering):
   ```bash
   pip install "mani_skill[rl]" sapien==3.0.0
   ```
3. Smoke-test the reference task:
   ```bash
   python -m mani_skill.examples.demo --env-id PickCube-v1 --render
   ```
   You should see the SO-100 arm pick up the default cube ~50% of the time with the scripted controller.

## 2. Assets you already have for SO-101
- URDF: `announce/assets/SO101/so101.urdf`
- SRDF: `announce/assets/SO101/so101.srdf`
- Meshes: `announce/assets/SO101/assets/*.stl`
Copy these into your ManiSkill data path (defaults to `~/.mani_skill/`), or point to them explicitly via an absolute path when creating the robot config.

## 3. Mapping SO-100 → SO-101
| Component | ManiSkill SO-100 default | Required SO-101 setting |
| --- | --- | --- |
| DOF | 6 revolute joints + 1 gripper slider | 6 revolute joints + 1 wrist roll + gripper, joint names from `so101.urdf` |
| Base pose | Fixed at world origin, Z=0.0 | Offset by the real table height (0.74 m by default in Track 1 scenes) |
| End-effector | `tool0` frame | `wrist_link` → `gripper_link` chain (see URDF) |
| Joint limits | `[-2.62, 2.62]` rad wide symmetric | Use `<limit>` tags in `so101.urdf`; several joints are asymmetric |
| Default controller | PD joint delta | Keep PD but update gains to reflect lighter joints (e.g., `kp=4.0`, `kd=0.4`) |

## 4. Implementing the SO-101 robot in ManiSkill
Create a minimal wrapper that swaps the robot config while leaving the task logic intact.

```python
# file: announce/maniskill_so101_env.py
import gym
import mani_skill.envs  # register envs
from pathlib import Path

ASSET_ROOT = Path(__file__).resolve().parents[1] / "announce" / "assets" / "SO101"

SO101_ROBOT_CFG = dict(
    arm_type="so101",
    urdf_path=str(ASSET_ROOT / "so101.urdf"),
    srdf_path=str(ASSET_ROOT / "so101.srdf"),
    root_link="base_link",
    ee_link="gripper_link",
    joint_names=[
        "shoulder_joint", "upper_arm_joint", "elbow_joint",
        "lower_arm_joint", "wrist_pitch_joint", "wrist_roll_joint",
        "gripper_joint"
    ],
)

def make_env(obs_mode="rgbd", control_mode="pd_joint_delta_pos"):
    env = gym.make(
        "PickCube-v1",
        obs_mode=obs_mode,
        control_mode=control_mode,
        robot_cfg=SO101_ROBOT_CFG,
        workspace_half_size=(0.25, 0.25, 0.25),  # matches Track 1 table
        table_height=0.74,
    )
    return env
```

Notes:
- `robot_cfg` is supported out-of-the-box in ManiSkill ≥3.0.1. For ManiSkill2, subclass `PickCubeEnv` and override `DEFAULT_ROBOT_CFG` instead.
- Keep the SRDF handy if you want auto-collision filtering; Sapien will parse it when provided via `robot_cfg`.

## 5. Scene and observation adjustments
1. **Workspace bounds** – The Track 1 bins are 30×30 cm. Adjust `workspace_half_size` accordingly.
2. **Table and bin meshes** – Import the real-scene meshes (once they are released) via `scene_builder.add_articulation(...)` so camera rays match reality.
3. **Cameras** – Mirror the real front/wrist intrinsics:
   ```python
   env.unwrapped.set_camera(
       name="wrist",
       pose=((0.0, 0.0, 0.06), (0.0, 0.0, 0.0, 1.0)),
       width=320,
       height=240,
       fov=70.0,
   )
   ```
4. **Action scaling** – The real interface expects joint deltas in radians and gripper commands in [0, 1]. Match this when writing the policy wrapper.

## 6. Validation checklist
- `python announce/tests/check_maniskill_so101.py --headless` (script stub below) should:
  - Reset the env with deterministic seeds.
  - Roll out 20 scripted trajectories.
  - Assert >90% grasp success with the default cube & table height.
- Visual inspection: run the same script with `--render` and verify the cameras match the screenshots in `announce/front_camera.png`.

Example stub:
```python
# announce/tests/check_maniskill_so101.py
from announce.maniskill_so101_env import make_env
from mani_skill.utils.wrappers import RecordEpisode

def main(episodes=20, render=False):
    env = make_env(obs_mode="rgbd", control_mode="pd_joint_delta_pos")
    env = RecordEpisode(env, out_dir="/tmp/so101_maniskill_rollouts")
    success = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            action = env.unwrapped.controller.compute_reference_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        success += info.get("success", False)
    rate = success / episodes
    assert rate > 0.9, f"Success rate only {rate:.2f}"
    env.close()

if __name__ == "__main__":
    main()
```

## 7. Feeding data into LeRobot
Once the environment is validated, log rollouts in the LeRobot dataset format so the same training stack can be reused for real data.
1. Wrap the environment with the `lerobot.common.wrappers.GymEnvWrapper`.
2. Use `lerobot.collectors.teleop` or a simple policy to record episodes into `dataset/eai/<task>/data`. Store RGB wrist/front images plus proprioception (joint positions, velocities, gripper state).
3. Update `lerobot/examples/dataset/config.yaml` with a new entry, e.g. `so101_pick_cube_sim` pointing to the folder above.
4. Run `python -m lerobot.examples.dataset.convert --dataset so101_pick_cube_sim` to make sure the metadata matches the schema used by the Track 1 release.

## 8. Next steps
- Plug in your training method of choice (BC, ACT, π0, TD-MPC2, etc.) using the simulated dataset.
- Once the real-scene assets are released, swap the table/bin meshes and recalibrate the cameras using `announce/calibration.py` to minimize sim-to-real gaps.
- Schedule a TA check only after the scripted validation succeeds; bring the dataset and the policy weights for faster on-site verification.
