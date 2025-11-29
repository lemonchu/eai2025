The provided scripts are about real robot.

- robot.py: Tests the loading of the SO-101 simulation and the wrist camera setup.

- front_camera.py: Defines the front camera settings within the simulation.

- calibration.py: Determines the intrinsic parameters for the real robot's front camera.

- undistort.py: Corrects distortion and analyzes images taken by the real robot.

- distort.py: Adds distortion to the simulated pinhole camera model.

- robots: Stores joint calibration parameters for mapping between the real robot and the simulation.

- dummy_eval.py: Serves as a reference interface for real robot testing.

**These scripts are provided only for context and reference. You are free to choose any algorithms, simulation methods, or platforms for your actual work.**

## New: LeRobot-ready SO-101 scene

- `gym_so101_scene/`: Lightweight Gymnasium environment backed by Sapien. It exposes front + wrist RGB and joint states, mirroring the observation schema used by LeRobot.
- `run_so101_lerobot_demo.py`: Uses `lerobot.envs.factory.make_env_config` to instantiate the scene, roll out random actions, and save RGB frames. Example (headless):

	```bash
	cd /cephfs/hp/cm_projects/eai
	PYOPENGL_PLATFORM=egl python scene/run_so101_lerobot_demo.py --episodes 3 --record-dir outputs/so101_demo
	```

	Each episode folder contains `front/` and `wrist/` PNGs. Convert to MP4 with

	```bash
	ffmpeg -y -framerate 30 \
		-pattern_type glob -i 'outputs/so101_demo/episode_000/front/*.png' \
		outputs/so101_demo/episode_000_front.mp4
	```

- The environment is registered as `gym_so101_scene/ScenePickCube-v0`, and `lerobot` can now reference it through `make_env_config("so101_scene", ...)` for training or evaluation workflows.

### Layout data from Track-1 measurements

- `gym_so101_scene/layout.py` encodes the centimeter measurements from Figure 2 (table 60×80 cm, boundary width 1.8 cm, robot bases, camera pose at (31.6 cm, 15.4 cm, 40.7 cm)).
- `TableLayout` converts those values to meters and spawns the boundary lines, colored robot-base markers, and camera mount in Sapien when the env resets.
- Adjust `TableLayout.world_origin` if you need to shift the entire tabletop relative to the robot base; tweak the `horizontal_lines_cm` / `vertical_lines_cm` tuples to match new measurements.