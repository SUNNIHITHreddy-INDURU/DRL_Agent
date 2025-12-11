import highway_env
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  

def make_env(obs_type):
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
    return env


def evaluate(model_path, obs_type, plot_path):

    env = make_env(obs_type)
    model = PPO.load(model_path)

    rewards = []

    print(f"\n🔍 Evaluating {obs_type}...")

    # Use tqdm progress bar
    for _ in tqdm(range(500), desc=f"Evaluating {obs_type}"):
        done = False
        total_reward = 0

        obs, _ = env.reset()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    # Save violin plot
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=rewards)
    plt.title(f"Performance Reward Distribution ({obs_type})")
    plt.ylabel("Episode Reward")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved performance plot: {plot_path}\n")


if __name__ == "__main__":

    # ID 2: Lidar
    evaluate(
        model_path="models/highway_lidar",
        obs_type="LidarObservation",
        plot_path="plots/2_performance_highway_lidar.png"
    )

    # ID 4: Grayscale
    evaluate(
        model_path="models/highway_gray",
        obs_type="GrayscaleObservation",
        plot_path="plots/4_performance_highway_gray.png"
    )
