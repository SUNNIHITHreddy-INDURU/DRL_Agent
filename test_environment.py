"""
Quick environment testing script
"""

import gymnasium as gym
import highway_env
import numpy as np
from multi_stage_env import HighwayConstructionEnv


def quick_visual_test():
    print("\n" + "=" * 60)
    print("VISUAL TEST (Press Ctrl+C to stop)")
    print("=" * 60)

    try:
        env = gym.make("highway-construction-v0", render_mode="human")
        obs, info = env.reset()

        print("Environment loaded successfully. Rendering...")

        for step in range(500):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if terminated or truncated:
                print("Episode ended, resetting...")
                obs, info = env.reset()

        env.close()

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    quick_visual_test()
