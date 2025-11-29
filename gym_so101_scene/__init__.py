from gymnasium.envs.registration import register

# Register a single environment id mapping to the local So101SceneEnv.
# Keep the registration minimal and avoid importing mani_wrapper here so
# layout-based changes in `gym_so101_scene.layout` take effect immediately
# when the environment is created by the user script.
register(
    id="LiftCubeSO101-v0",
    entry_point="gym_so101_scene.env:So101SceneEnv",
    max_episode_steps=300,
)

__all__ = ["So101SceneEnv"]
