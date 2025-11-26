import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# ------------------------------------------
# Training model
# ------------------------------------------

def make_env():
    env = gym.make("merge-v0", render_mode=None)
    # Environment configuration:
    # Action Space: Mode A (Discrete)
    # LidarObservation. 
    env.unwrapped.configure({
        "action": {
            "type": "DiscreteMetaAction",
        },
        "observation": {
            "type": "LidarObservation",
            "cells": 128, # default values
            "maximum_range": 64, # default values
            "normalize": True
        }
    })
    env.reset()
    return env

# Create log dir
log_dir = "./logs/lidar/"
os.makedirs(log_dir, exist_ok=True)

# Initialize environment and agent
# Wrap the environment with Monitor to save training data
# Logging: Wrap the environment with stable_baselines3.common.monitor.Monitor
env = Monitor(make_env(), filename=os.path.join(log_dir, "monitor.csv"))

# Preprocessing: Wrap in DummyVecEnv.
env = DummyVecEnv([lambda: env])

# Algorithm: DQN with MlpPolicy
# Architecture: Custom net_arch=[256, 256](Handle Multi-car interactions)
# SB3 Documentation: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html#parameters
model = DQN(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1
)

print("Starting training for Lidar Agent...")
model.learn(total_timesteps=20000)
print("Training finished.")

# Save the model
model_path = os.path.join(log_dir, "lidar_dqn_model")
model.save(model_path)
print(f"Model saved to {model_path}")

# Close training env
env.close()

# ------------------------------------------
# Visualization 
# ------------------------------------------

print("\nStarting Visualization...")

# Create a new test environment with render_mode='human'
viz_env = gym.make("merge-v0", render_mode='human')
viz_env.unwrapped.configure({
    "action": {
        "type": "DiscreteMetaAction",
    },
    "observation": {
        "type": "LidarObservation",
        "cells": 128,
        "maximum_range": 64,
        "normalize": True
    }
})

# Load the saved model
loaded_model = DQN.load(model_path)

# Run a loop for 3 episodes
for episode in range(3):
    obs, info = viz_env.reset()
    done = truncated = False
    total_reward = 0
    while not (done or truncated):
        action, _ = loaded_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = viz_env.step(action)
        total_reward += reward
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

viz_env.close()

# ------------------------------------------
# Eval and Plotting
# ------------------------------------------

def plot_results(log_folder):
    """
    Read monitor.csv and plot Training Steps vs. Mean Reward (Learning Curve)
    """
    monitor_path = os.path.join(log_folder, "monitor.csv")
    if not os.path.exists(monitor_path):
        print(f"No monitor.csv found at {monitor_path}")
        return

    # Skip first line
    df = pd.read_csv(monitor_path, skiprows=1)
    
    # Calculate cumulative timesteps
    df['timesteps'] = df['l'].cumsum()
    
    # Rolling average smoothing
    window_size = 50
    df['reward_smooth'] = df['r'].rolling(window=window_size).mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['timesteps'], df['r'], alpha=0.3, label='Raw Reward')
    plt.plot(df['timesteps'], df['reward_smooth'], label=f'Smoothed Reward (window={window_size})', color='blue')
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('Lidar Agent Learning Curve')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(log_folder, "learning_curve.png")
    plt.savefig(save_path)
    print(f"Learning curve saved to {save_path}")
    plt.close()

def plot_violin(model, env_config_fn):
    """
    Run evaluate_policy and create a Violin Plot of the returns
    """
    # Create a fresh evaluation env
    eval_env = Monitor(env_config_fn())
    eval_env = DummyVecEnv([lambda: eval_env])
    
    print("Running evaluation for Violin Plot...")
    # return_episode_rewards=True is crucial for getting the list of rewards
    episode_rewards, episode_lengths = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=100, 
        return_episode_rewards=True
    )
    
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=episode_rewards, inner="quartile")
    plt.title('Distribution of Episode Rewards (Lidar Agent)')
    plt.ylabel('Reward')
    plt.xlabel('Lidar Agent')
    
    save_path = os.path.join(log_dir, "violin_plot.png")
    plt.savefig(save_path)
    print(f"Violin plot saved to {save_path}")
    plt.close()
    eval_env.close()

# Execute plotting functions
plot_results(log_dir)
plot_violin(loaded_model, make_env)
