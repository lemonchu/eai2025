For details about the legacy real-robot helper scripts, see `scene/README.md`.

## New: LeRobot-ready SO-101 scene

- `gym_so101_scene/`: Lightweight Gymnasium environment backed by Sapien. It now spawns two SO-101 follower arms and exposes front + left/right wrist RGB along with per-arm joint states, mirroring the observation schema used by LeRobot.
- `run_so101_lerobot_demo.py`: Uses `lerobot.envs.factory.make_env_config` to instantiate the scene, roll out random actions, and save RGB frames. Example (headless):

	```bash
	cd /cephfs/hp/cm_projects/eai/eai2025
	PYOPENGL_PLATFORM=egl python run_so101_lerobot_demo.py --episodes 3 --record-dir outputs/so101_demo
	```

	Each episode folder contains `front/`, `left_wrist/`, and `right_wrist/` PNGs. Convert to MP4 with

	```bash
	ffmpeg -y -framerate 30 \
		-pattern_type glob -i 'outputs/so101_demo/episode_000/front/*.png' \
		outputs/so101_demo/episode_000_front.mp4
	```

- The environment is registered as `gym_so101_scene/ScenePickCube-v0`, and `lerobot` can now reference it through `make_env_config("so101_scene", ...)` for training or evaluation workflows.

### Layout data from Track-1 measurements

- `gym_so101_scene/layout.py` encodes the centimeter measurements from Figure 2:
  - **Total table size**: 120.0 cm × 60.0 cm (left area 60.0 cm + right area 58.2 cm + boundary width 1.8cm)
  - **Vertical boundary line**: at x=60.0 cm (1.8 cm wide, separates left/right working areas)
  - **Three bins at top**: widths [16.6, 15.6, 16.6] cm, height 16.4 cm
  - **Robot areas at bottom**: width 20.4 cm, depth 15.0 cm
  - **Robot base offset**: 6.4 cm from edge, size 11.0 × 8.1 cm
  - **Camera pose**: (31.6 cm, 26.0 cm, 40.7 cm) from layout origin
- `TableLayout` converts those values to meters, spawns the black boundary lines, robot area markers, and positions both robot-base markers + cameras in Sapien when the env resets.
- Adjust `TableLayout.world_origin` if you need to shift the entire tabletop relative to the robot bases; tweak the convenience helpers (e.g., `bin_widths_cm`, `robot_base_offset_cm`) to match future measurements.