import argparse
import gym_so101_scene  # ensure local package is imported so env registrations run
import gymnasium as gym
from PIL import Image


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--task", type=str, default="lift", help="Task to spawn: lift|stack|sort")
	args = parser.parse_args()

	env = gym.make("LiftCubeSO101-v0", render_mode="rgb_array", headless=True, task=args.task)
	obs, info = env.reset()
	print("obs_compat:", env.unwrapped.get_obs_compat())
	rgba = env.unwrapped.take_picture("front")
	out = f"front_view_{args.task}.png"
	Image.fromarray(rgba).save(out)
	print("Saved", out)
	print("Success:", env.unwrapped.check_success())
	env.close()


if __name__ == "__main__":
	main()