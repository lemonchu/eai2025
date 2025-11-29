from gymnasium.envs.registration import register

register(
    id="gym_so101_scene/ScenePickCube-v0",
    entry_point="gym_so101_scene.env:So101SceneEnv",
)

__all__ = ["So101SceneEnv"]
