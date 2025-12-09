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
    env = gym.make("intersection-v0", render_mode=None)
    # GrayscaleObservation with stack_size=4
    env.unwrapped.configure({
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,  # stack size for images
            "weights": [0.2989, 0.5870, 0.1140],  # Standard RGB to Grayscale weights
            "scaling": 1.75,
        },
        "action": {
            "type": "DiscreteMetaAction",
        }
    })
    env.reset()
    return env


# Create log dir
log_dir = "./logs/greyscale/"
os.makedirs(log_dir, exist_ok=True)

env = Monitor(make_env(), filename=os.path.join(log_dir, "monitor.csv"))

env = DummyVecEnv([lambda: env])

model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=200000,
    learning_starts=5000,
    batch_size=64,
    train_freq=4,
    target_update_interval=2000,
    exploration_fraction=0.3,
    exploration_final_eps=0.01,
    gamma=0.99,
    max_grad_norm=10,
    verbose=1
)


print("Starting training for Grayscale Agent...")
model.learn(total_timesteps=20000)  # action steps
print("Training finished.")

# Save the model
model_path = os.path.join(log_dir, "greyscale_intersection_dqn_model")
model.save(model_path)
print(f"Model saved to {model_path}")

# Close training env
env.close()

# ------------------------------------------
# Visualization
# ------------------------------------------

print("\nStarting Visualization...")

viz_env = gym.make("intersection-v0", render_mode='human')
viz_env.unwrapped.configure({
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],
        "scaling": 1.75,
    },
    "action": {
        "type": "DiscreteMetaAction",
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
    window_size = 50  # avg of last 50 timesteps
    df['reward_smooth'] = df['r'].rolling(window=window_size).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(df['timesteps'], df['r'], alpha=0.3, label='Raw Reward')
    plt.plot(df['timesteps'], df['reward_smooth'], label=f'Smoothed Reward (window={window_size})', color='blue')
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('Grayscale Agent Learning Curve')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(log_folder, "intersection_learning_curve.png")
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

    print("Running evaluation for Intersection Violin Plot...")
    # return_episode_rewards=True is crucial for getting the list of rewards
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=100,
        return_episode_rewards=True
    )

    plt.figure(figsize=(8, 6))
    sns.violinplot(data=episode_rewards, inner="quartile")
    plt.title('Distribution of Episode Rewards (Intersection Grayscale Agent)')
    plt.ylabel('Reward')
    plt.xlabel('Grayscale Agent')

    save_path = os.path.join(log_dir, "intersection_violin_plot.png")
    plt.savefig(save_path)
    print(f"Violin plot saved to {save_path}")
    plt.close()
    eval_env.close()


# Execute plotting functions
plot_results(log_dir)
plot_violin(loaded_model, make_env)
