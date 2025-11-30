import gymnasium as gym
import highway_env
import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# ------------------------------------------
# Visual view of agents
# ------------------------------------------

def visualize_lidar_agent():
    print("\n--- Visualizing Lidar Agent ---")
    
    def make_viz_env():
        # Create the environment with render_mode='human' to see it
        env = gym.make("merge-v0", render_mode='human')
        
        # Configure it EXACTLY like the training env
        env.unwrapped.configure({
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
        env.reset()
        return env

    # Path to the saved model
    model_path = "./logs/merge_lidar/lidar_dqn_model"

    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model not found at {model_path}.zip")
        print("Please run 'python MergeEnv/lidar_agent.py' first to train the agent.")
        return

    print(f"Loading model from {model_path}...")

    # init environment
    viz_env = Monitor(make_viz_env())
    viz_env = DummyVecEnv([lambda: viz_env]) 

    # Load the model
    model = DQN.load(model_path, env=viz_env)

    # # of episodes to run
    n_episodes = 20
    print(f"Running for {n_episodes} episodes...")

    for episode in range(n_episodes):
        obs = viz_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = viz_env.step(action)
            total_reward += reward
            
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    print("Lidar Visualization finished.")
    viz_env.close()


def visualize_grayscale_agent():
    print("\n--- Visualizing Grayscale Agent ---")

    def make_viz_env():
        env = gym.make("merge-v0", render_mode='human')
        # GrayscaleObservation with stack_size=4
        env.unwrapped.configure({
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 64),
                "stack_size": 4, # stack size for images
                "weights": [0.2989, 0.5870, 0.1140],  # Standard RGB to Grayscale weights
                "scaling": 1.75,
            },
            "action": {
                "type": "DiscreteMetaAction",
            }
        })
        env.reset()
        return env

    # Path to the saved model
    model_path = "./logs/merge_greyscale/greyscale_dqn_model"

    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model not found at {model_path}.zip")
        print("Please run 'python MergeEnv/greyscale_agent.py' first to train the agent.")
        return

    print(f"Loading model from {model_path}...")

    # init environment
    viz_env = Monitor(make_viz_env())
    viz_env = DummyVecEnv([lambda: viz_env]) 

    # Load the model
    model = DQN.load(model_path, env=viz_env)

    # # of episodes to run
    n_episodes = 20
    print(f"Running for {n_episodes} episodes...")

    for episode in range(n_episodes):
        obs = viz_env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = viz_env.step(action)
            total_reward += reward
            
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    print("Grayscale Visualization finished.")
    viz_env.close()

# main function to run either lidar or grayscale agent
if __name__ == "__main__":
    
    
    # visualize_lidar_agent()
    visualize_grayscale_agent()
