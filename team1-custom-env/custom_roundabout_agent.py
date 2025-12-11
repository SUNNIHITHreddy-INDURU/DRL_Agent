import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import highway_env
from register_envs import *

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


# ------------------------------------------
# Environment
# ------------------------------------------

def make_env():
    env = gym.make("custom-roundabout-v0")
    env.reset()
    return env


# ------------------------------------------
# Logging directory
# ------------------------------------------

log_dir = "./logs/roundabout/"
os.makedirs(log_dir, exist_ok=True)

tb_log = "./logs/tensorboard/roundabout/"


# ------------------------------------------
# Training
# ------------------------------------------

env = Monitor(make_env(), filename=os.path.join(log_dir, "monitor.csv"))
env = DummyVecEnv([lambda: env])

model = PPO(
    "MlpPolicy",
    env,
    n_steps=4096,              # collect more experience per update
    batch_size=512,            # more stable gradient updates
    n_epochs=15,               # more learning per batch
    learning_rate=2.5e-4,      # slightly more stable learning
    max_grad_norm=0.7,         # less aggressive clipping
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.15,           # tighter PPO trust region
    ent_coef=0.02,             # stronger exploration incentive
    vf_coef=0.5,
    verbose=1
)

print("Starting training for Custom Roundabout Agent...")
model.learn(
    total_timesteps=20000,
    # tb_log_name="PPO_Roundabout_Run1"
)
print("Training finished.")


# ------------------------------------------
# Save Model
# ------------------------------------------

model_path = os.path.join(log_dir, "roundabout_ppo_model")
model.save(model_path)
print(f"Model saved to {model_path}")

env.close()


# ------------------------------------------
# Visualization
# ------------------------------------------

print("\nStarting Visualization...")

viz_env = gym.make("custom-roundabout-v0", render_mode="human")
loaded_model = PPO.load(model_path)

for episode in range(3):
    obs, info = viz_env.reset()
    done = truncated = False
    total_reward = 0.0

    while not (done or truncated):
        action, _ = loaded_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = viz_env.step(action)
        total_reward += reward

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

viz_env.close()


# ------------------------------------------
# Evaluation + Plots
# (unchanged style from your other files)
# ------------------------------------------

def plot_results(log_folder):
    monitor_path = os.path.join(log_folder, "monitor.csv")
    if not os.path.exists(monitor_path):
        print(f"No monitor.csv found at {monitor_path}")
        return

    df = pd.read_csv(monitor_path, skiprows=1)

    df["timesteps"] = df["l"].cumsum()
    window_size = 50
    df["reward_smooth"] = df["r"].rolling(window=window_size).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(df["timesteps"], df["r"], alpha=0.3, label="Raw Reward")
    plt.plot(df["timesteps"], df["reward_smooth"], label="Smoothed Reward", color="blue")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("Roundabout Agent Learning Curve")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(log_folder, "learning_curve.png")
    plt.savefig(save_path)
    plt.close()


def plot_violin(model, env_fn):
    eval_env = Monitor(env_fn())
    eval_env = DummyVecEnv([lambda: eval_env])

    episode_rewards, _ = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=100,
        return_episode_rewards=True
    )

    plt.figure(figsize=(8, 6))
    sns.violinplot(data=episode_rewards, inner="quartile")
    plt.title("Distribution of Episode Rewards (Roundabout Agent)")
    plt.ylabel("Reward")
    plt.xlabel("Roundabout Agent")

    save_path = os.path.join(log_dir, "violin_plot.png")
    plt.savefig(save_path)
    plt.close()
    eval_env.close()


plot_results(log_dir)
plot_violin(loaded_model, make_env)
