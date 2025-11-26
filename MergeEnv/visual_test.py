import gymnasium as gym
import highway_env
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# ------------------------------------------
# Visual view of lidar agent
# ------------------------------------------

def make_viz_env():
    # Create the environment with render_mode='human' to see it
    env = gym.make("merge-v0", render_mode='human')
    
    # Configure it EXACTLY like the training env to ensure model compatibility
    # IMPORTANT: Use env.unwrapped.configure() to avoid AttributeError
    # Configuration matches User Request for Lidar Agent:
    # "Observation: LidarObservation... 16 beams... max range of 32 meters"
    env.unwrapped.configure({
        "action": {
            "type": "DiscreteMetaAction",
        },
        "observation": {
            "type": "LidarObservation",
            "cells": 16,
            "maximum_range": 32,
            "normalize": True
        }
    })
    env.reset()
    return env

# Path to the saved model
# Note: SB3 adds .zip automatically, so we just point to the base name
model_path = "./logs/lidar/lidar_dqn_model"

if not os.path.exists(model_path + ".zip"):
    print(f"Error: Model not found at {model_path}.zip")
    print("Please run 'python MergeEnv/lidar_agent.py' first to train the agent.")
    exit(1)

print(f"Loading model from {model_path}...")

# init environment
# Monitor is used for watching agent
# Wrap in DummyVecEnv to follow SB3 documentation
viz_env = Monitor(make_viz_env())
viz_env = DummyVecEnv([lambda: viz_env]) # Follow Vectorized Environment

# Load the model
# SB3 Documentation: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html#stable_baselines3.dqn.DQN.load
model = DQN.load(model_path, env=viz_env)

# Run for a few episodes
n_episodes = 20 
print(f"Running for {n_episodes} episodes...")

for episode in range(n_episodes):
    obs = viz_env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Predict action 
        action, _ = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, done, info = viz_env.step(action)
        
        total_reward += reward
        
        # Render is handled with ENV: render_mode='human'
        
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

print("Visualization finished.")
viz_env.close()
