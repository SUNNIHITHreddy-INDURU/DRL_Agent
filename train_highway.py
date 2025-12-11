import highway_env
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import os

def make_env(obs_type="LidarObservation"):

    if obs_type == "GrayscaleObservation":
        obs_config = {
            "type": "GrayscaleObservation",
            "observation_shape": (84, 84),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": 1.75
        }
    else:
        obs_config = {"type": obs_type}

    config = {
        "observation": obs_config,
        "policy_frequency": 15
    }

    env = gym.make("highway-v0", render_mode=None, config=config)
    env = Monitor(env)
    return env


def train(obs_type, model_path, log_dir):

    env = make_env(obs_type)

    print(f"\n==============================")
    print(f" STARTING TRAINING: {obs_type}")
    print(f"==============================\n")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        batch_size=64,
        n_steps=2048
    )

    logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)

    model.learn(total_timesteps=200000)

    model.save(model_path)

    print(f"\n==============================")
    print(f" FINISHED TRAINING: {obs_type}")
    print(f" Saved model to {model_path}")
    print(f"==============================\n")


if __name__ == "__main__":

    # === CONTROL WHAT TO TRAIN ===
    TRAIN_LIDAR = True
    TRAIN_GRAYSCALE = True


    if TRAIN_LIDAR:
        train(
            obs_type="LidarObservation",
            model_path="models/highway_lidar",
            log_dir="logs/highway_lidar"
        )

 
    if TRAIN_GRAYSCALE:
        train(
            obs_type="GrayscaleObservation",
            model_path="models/highway_gray",
            log_dir="logs/highway_gray"
        )
