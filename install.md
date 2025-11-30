# EAI Track 1 Environment Setup

This guide walks you through preparing a headless Linux server for Track 1 experiments using the TA-provided SO-101 assets plus ManiSkill/Gym tooling. Commands assume Ubuntu 20.04+/22.04+ with bash and sudo access.

## 1. System Requirements

- Ubuntu server with an NVIDIA GPU + recent driver (verify via `nvidia-smi` and `vulkaninfo`).
- Conda (Miniconda/Anaconda). Install from https://docs.conda.io if not present.
- Basic build and graphics libraries:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git \
  libgl1-mesa-dev libegl1-mesa-dev \
  libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev \
  ffmpeg
```

> If you lack sudo, coordinate with the admin to install the packages once. ManiSkill will attempt CPU/EGL fallbacks but the packages avoid most Sapien/Vulkan errors.

## 2. Create an Isolated Conda Environment

```bash
cd /cephfs/hp/cm_projects/eai
conda create -n eai2025 python=3.10 -y
conda activate eai2025
pip install --upgrade pip
```

Keep the environment activated (`conda activate eai2025`) whenever working on the project.

## 3. Install ManiSkill + Rendering Dependencies

```bash
pip install "mani-skill>=3.0.0" imageio[ffmpeg] opencv-python
```

If PyPI is slow, append `-i https://pypi.tuna.tsinghua.edu.cn/simple`.

## 4. Install Gym / Gymnasium Tooling

```bash
pip install "gymnasium[classic-control]" "gym==0.26.2" matplotlib
```

The dual install keeps compatibility with older configs that still import `gym`.

## 5. Quick Sanity Checks

### 5.1 Gym CartPole (CPU-only)

```bash
python - <<'PY'
import gymnasium as gym
env = gym.make("CartPole-v1")
obs, info = env.reset()
for _ in range(10):
	action = env.action_space.sample()
	obs, reward, terminated, truncated, info = env.step(action)
	if terminated or truncated:
		obs, info = env.reset()
env.close()
print("Gym CartPole OK")
PY
```

### 5.2 ManiSkill PickCube Headless

```bash
python - <<'PY'
import gymnasium as gym
import mani_skill.envs

env = gym.make(
	"PickCube-v1",
	obs_mode="state",
	render_mode="none",
)
obs, info = env.reset(seed=0)
for _ in range(20):
	action = env.action_space.sample()
	obs, reward, terminated, truncated, info = env.step(action)
	if terminated or truncated:
		obs, info = env.reset()
env.close()
print("ManiSkill PickCube headless OK")
PY
```

## 6. (Optional) Generate a Reference Video

Use `gen_so100_video.py` at the repository root to confirm RGB rendering:

```bash
conda activate eai2025
cd /cephfs/hp/cm_projects/eai
python gen_so100_video.py
```

It produces `so100_pickcube_demo.mp4`. Inspect the clip to ensure the SO-100 pick-and-place scene matches expectations.

## 7. Next Steps

- Plug in your datasets under `dataset/eai/*` using your preferred processors.
- Extend ManiSkill’s SO-100 task description to SO-101 by swapping URDF/SRDF files inside `scene/assets/SO101/`.
- Version-control any new scripts (env wrappers, training configs) under `policies/` or another folder in this repository so everything stays self-contained.

## 8. Laptop (With Display) Tips

If you want to run the same stack on a personal laptop for debugging or visualization:

1. **Graphics Drivers**: Ensure the laptop uses NVIDIA/AMD proprietary drivers or a recent Mesa stack. On Ubuntu, `sudo ubuntu-drivers autoinstall` usually suffices.
2. **Use Micromamba/Miniconda**: Resource-constrained laptops benefit from Micromamba’s lower overhead. Create the same `eai2025` environment so configs remain consistent across machines.
3. **OpenGL GUI Rendering**: Launch ManiSkill envs with `render_mode="human"` to pop up interactive windows:

   ```bash
   python - <<'PY'
   import gymnasium as gym
   import mani_skill.envs

   env = gym.make(
	   "PickCube-v1",
	   obs_mode="rgbd",
	   render_mode="human",
   )
   env.reset()
   for _ in range(200):
	   obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
	   if terminated or truncated:
		   env.reset()
   env.close()
   PY
   ```

4. **VS Code Remote or Jupyter**: Use VS Code Remote SSH to keep compute on the server while forwarding GUI windows to the laptop (via `ssh -X`) for quick visual inspections.
5. **Battery Considerations**: Limit FPS (`imageio.mimwrite(..., fps=15)`) and frame counts when capturing videos locally to avoid thermal throttling.

With these steps complete, the server is ready for Track 1 development and reproducible runs.
