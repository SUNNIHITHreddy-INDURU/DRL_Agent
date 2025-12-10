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

# Create log dir
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs/merge_greyscale/")
os.makedirs(log_dir, exist_ok=True)

# init environment and agent
# Wrap the environment with Monitor to save training data
env = Monitor(make_env(), filename=os.path.join(log_dir, "monitor.csv"))

# Wrap in DummyVecEnv.
env = DummyVecEnv([lambda: env])

#  DQN with CnnPolicy
model = DQN(
    "CnnPolicy",
    env,
    buffer_size=100000, # 1/10 of default for 8gb ram @ moment -- change for larger ram --
    learning_rate=1e-4,
    batch_size=32,             # reduce batch size
    gamma=0.98,                # lower discount 
    train_freq=4,              # Train every 4 steps
    gradient_steps=1,          # 1 gradient step per training trigger
    target_update_interval=2000, # Update target net more often (default 10K)
    learning_starts=2000,      # start learning at 2K steps
    exploration_fraction=0.3,  # Explore for 30% of total timesteps (longer exploration)
    exploration_final_eps=0.05, # keep 5% exploration at all times after
    verbose=1
)

print("Starting training for Grayscale Agent...")
model.learn(total_timesteps=200000) # action steps
print("Training finished.")

# Save the model
model_path = os.path.join(log_dir, "greyscale_dqn_model")
model.save(model_path)
print(f"Model saved to {model_path}")

# Close training env
env.close()

# ------------------------------------------
# Visualization 
# ------------------------------------------

print("\nStarting Visualization...")

# new test environment with render_mode='human'
# final shape (4, 128, 64), depth, width, height
viz_env = gym.make("merge-v0", render_mode='human')
viz_env.unwrapped.configure({
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64), # width, height agent see. View is longer towards lanes
        "stack_size": 4, # current frame + 3 previous frames
        "weights": [0.2989, 0.5870, 0.1140],
        "scaling": 1.75, # 1.75Pixels per Meter(zoomed out view)
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
    window_size = 50 # avg of last 50 timesteps
    df['reward_smooth'] = df['r'].rolling(window=window_size).mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['timesteps'], df['r'], alpha=0.3, label='Raw Reward')
    plt.plot(df['timesteps'], df['reward_smooth'], label=f'Smoothed Reward (window={window_size})', color='blue')
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Reward')
    plt.title('Grayscale Agent Learning Curve')
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
    plt.title('Distribution of Episode Rewards (Grayscale Agent)')
    plt.ylabel('Reward')
    plt.xlabel('Grayscale Agent')
    
    save_path = os.path.join(log_dir, "violin_plot.png")
    plt.savefig(save_path)
    print(f"Violin plot saved to {save_path}")
    plt.close()
    eval_env.close()

# Execute plotting functions
plot_results(log_dir)
plot_violin(loaded_model, make_env)
