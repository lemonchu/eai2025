# Simple Pick Policy

This folder contains a scripted joint-space controller intended as a starting
point for SO-101 experiments. The policy relies purely on kinematic heuristics:

1. Move the right arm into a pre-grasp pose above the pick bin.
2. Descend until the end-effector is close to the cube.
3. Close the gripper.
4. Lift the cube straight up to an exit pose.

The implementation intentionally keeps the logic transparent, so you can swap in
learned components or inject perception throughout the finite-state machine.

## Usage

```bash
python -m policies.simple_pick.run_simple_pick_policy --episodes 1 --task lift
```

To turn the saved PNGs into an mp4 (one episode, front camera):

```bash
ffmpeg -y -framerate 30 -pattern_type glob -i 'outputs/simple_pick/episode_000/front/*.png' outputs/simple_pick_episode_000_front.mp4
```

Key flags:

- `--task`: pick between `lift`, `stack`, `sort` tasks defined by the scene.
- `--arm`: select `left` or `right` arm (right by default).
- `--record-dir`: directory where front-camera frames are dumped per episode.

The script mirrors `run_so101_lerobot_demo.py`: it builds the same environment
directly via `gym.make("gym_so101_scene/ScenePickCube-v0", ...)` using the
TA-provided assets under `scene/assets/SO101`, then repeatedly calls
`SimplePickPolicy.act` to produce delta joint commands.
