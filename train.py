import argparse
import os

import gymnasium as gym
import highway_env
import numpy as np

from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# QR-DQN from sb3-contrib
from sb3_contrib import QRDQN

# Import custom env file so registration side-effect runs
import multi_stage_env


# ---------------------------------------------------------
# ENV CREATION (supports lidar & grayscale)
# ---------------------------------------------------------
def make_env(env_name, obs_type="lidar", render_mode=None):
    """
    Create a highway-env / custom env and configure observation type.
    Handles Gymnasium wrappers by unwrapping until the base env.
    """
    env = gym.make(env_name, render_mode=render_mode)

    # Unwrap Gymnasium wrappers (OrderEnforcing, PassiveEnvChecker, etc.)
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    # Configure observation
    if obs_type.lower() == "lidar":
        base_env.configure({
            "observation": {
                "type": "LidarObservation",
                "cells": 128,
            }
        })

    elif obs_type.lower() == "grayscale":
        base_env.configure({
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],
            }
        })

    # Apply config
    env.reset()
    return env


# ---------------------------------------------------------
# MAP EXP-ID → ENV-NAME + OBSERVATION TYPE
# ---------------------------------------------------------
def get_env_name_from_id(exp_id: int):
    """
    Map the experiment ID (1–16) to (env_name, obs_type),
    matching the project spec.
    """
    env_map = {
        1: ("highway-v0", "lidar"),
        2: ("highway-v0", "lidar"),
        3: ("highway-v0", "grayscale"),
        4: ("highway-v0", "grayscale"),

        5: ("merge-v0", "lidar"),
        6: ("merge-v0", "lidar"),
        7: ("merge-v0", "grayscale"),
        8: ("merge-v0", "grayscale"),

        9: ("intersection-v0", "lidar"),
        10: ("intersection-v0", "lidar"),
        11: ("intersection-v0", "grayscale"),
        12: ("intersection-v0", "grayscale"),

        # Your custom env
        13: ("highway-construction-v0", "lidar"),
        14: ("highway-construction-v0", "lidar"),

        # Placeholder for other team's env
        15: ("other-team-env-v0", "lidar"),
        16: ("other-team-env-v0", "lidar"),
    }

    if exp_id not in env_map:
        raise ValueError(f"Unknown experiment ID: {exp_id}")

    return env_map[exp_id]


# ---------------------------------------------------------
# TRAIN FUNCTION
# ---------------------------------------------------------
def train(args):
    """Train DRL agent."""
    print("\n" + "=" * 80)
    print(f"TRAINING | Algo = {args.algo.upper()} | Exp-ID = {args.exp_id}")
    print(f"Env     : {args.env_name}")
    print(f"ObsType : {args.obs_type}")
    print(f"Steps   : {args.total_timesteps}")
    print("=" * 80 + "\n")

    # Directories
    log_dir = f"logs/exp_{args.exp_id}_{args.algo}_{args.obs_type}"
    model_dir = f"models/exp_{args.exp_id}"
    tensorboard_dir = f"tensorboard/exp_{args.exp_id}_{args.algo}_{args.obs_type}"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Create training environment
    def make_train_env():
        env = make_env(args.env_name, args.obs_type)
        return Monitor(env, log_dir)

    env = DummyVecEnv([make_train_env])

    # Eval env
    eval_env = DummyVecEnv([lambda: make_env(args.env_name, args.obs_type)])

    # Select algorithm
    algo_class = {
        "ppo": PPO,
        "dqn": DQN,
        "a2c": A2C,
        "sac": SAC,
        "qrdqn": QRDQN,
    }[args.algo.lower()]

    # --------- Hyperparameters per algorithm ---------
    if args.algo.lower() == "ppo":
        model_kwargs = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "tensorboard_log": tensorboard_dir,
            "verbose": 1,
        }

    elif args.algo.lower() == "dqn":
        model_kwargs = {
            "learning_rate": 1e-3,
            "buffer_size": 100_000,
            "learning_starts": 2_000,
            "batch_size": 64,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1_000,
            "exploration_fraction": 0.2,
            "exploration_final_eps": 0.05,
            "tensorboard_log": tensorboard_dir,
            "verbose": 1,
        }

    elif args.algo.lower() == "qrdqn":
        # Optimized QR-DQN for your custom env
        model_kwargs = {
            "learning_rate": 3e-4,          # smaller LR for stable quantile updates
            "buffer_size": 200_000,         # more replay diversity
            "learning_starts": 5_000,       # warm-up before learning
            "batch_size": 128,              # larger batch for stable gradients
            "gamma": 0.995,                 # long-term planning (construction zone etc.)
            "train_freq": (4, "step"),      # update every 4 steps
            "gradient_steps": 2,            # more gradient steps per update
            "target_update_interval": 2_000,
            "exploration_fraction": 0.25,   # more exploration early
            "exploration_final_eps": 0.02,  # more confident policy later
            "policy_kwargs": dict(
                net_arch=[256, 256],        # larger network for complex behavior
                n_quantiles=50,             # more quantiles → better distribution estimate
            ),
            "tensorboard_log": tensorboard_dir,
            "verbose": 1,
        }

    elif args.algo.lower() == "a2c":
        model_kwargs = {
            "learning_rate": 7e-4,
            "gamma": 0.99,
            "n_steps": 5,
            "ent_coef": 0.01,
            "tensorboard_log": tensorboard_dir,
            "verbose": 1,
        }

    else:  # SAC
        model_kwargs = {
            "learning_rate": 3e-4,
            "buffer_size": 200_000,
            "batch_size": 128,
            "gamma": 0.99,
            "tau": 0.005,
            "train_freq": 1,
            "gradient_steps": 1,
            "tensorboard_log": tensorboard_dir,
            "verbose": 1,
        }

    # Create model
    model = algo_class("MlpPolicy", env, **model_kwargs)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=model_dir,
        name_prefix=f"exp_{args.exp_id}_{args.algo}_{args.obs_type}",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=20_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Train
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    print(f"\nModel saved at: {final_path}\n")

    env.close()
    eval_env.close()


# ---------------------------------------------------------
# TEST FUNCTION
# ---------------------------------------------------------
def test(args):
    """Test a trained model (for performance evaluation / violin plot)."""
    print("\n" + "=" * 80)
    print(f"TESTING | Algo = {args.algo.upper()} | Exp-ID = {args.exp_id}")
    print(f"Env     : {args.env_name}")
    print(f"ObsType : {args.obs_type}")
    print(f"Model   : {args.load_model}")
    print("=" * 80 + "\n")

    env = make_env(args.env_name, args.obs_type, render_mode="human")

    algo_class = {
        "ppo": PPO,
        "dqn": DQN,
        "a2c": A2C,
        "sac": SAC,
        "qrdqn": QRDQN,
    }[args.algo.lower()]

    model = algo_class.load(args.load_model)

    rewards = []
    lengths = []

    for ep in range(args.test_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_r += r
            steps += 1
            env.render()

        rewards.append(total_r)
        lengths.append(steps)
        print(f"Episode {ep + 1}/{args.test_episodes}: Reward = {total_r:.2f}, Length = {steps}")

    print("\n" + "-" * 60)
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean Length: {np.mean(lengths):.2f}")
    print("-" * 60 + "\n")

    # Save results for plotting violin plot
    results_dir = f"results/exp_{args.exp_id}"
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"test_results_exp_{args.exp_id}.npz")
    np.savez(out_path,
             rewards=np.array(rewards),
             lengths=np.array(lengths),
             mean_reward=np.mean(rewards),
             std_reward=np.std(rewards))
    print(f"Saved test results to: {out_path}")

    env.close()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Highway / Custom Env DRL Training & Testing")

    parser.add_argument("--mode", choices=["train", "test"], required=True,
                        help="Train or test mode")
    parser.add_argument("--exp-id", type=int, required=True,
                        help="Experiment ID (1–16)")
    parser.add_argument("--algo", type=str, default="qrdqn",
                        choices=["ppo", "dqn", "a2c", "sac", "qrdqn"],
                        help="RL algorithm to use")
    parser.add_argument("--env-name", type=str, default=None,
                        help="Environment name (auto from exp-id if not set)")
    parser.add_argument("--obs-type", type=str, default=None,
                        choices=["lidar", "grayscale"],
                        help="Observation type (auto from exp-id if not set)")
    parser.add_argument("--total-timesteps", type=int, default=500_000,
                        help="Total training steps")
    parser.add_argument("--test-episodes", type=int, default=500,
                        help="Number of test episodes")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to trained model for test mode")

    args = parser.parse_args()

    # Auto-detect env + observation from experiment ID
    if args.env_name is None or args.obs_type is None:
        env_name, obs_type = get_env_name_from_id(args.exp_id)
        args.env_name = env_name
        args.obs_type = obs_type

    if args.mode == "train":
        train(args)
    else:
        if not args.load_model:
            raise ValueError("Must provide --load-model in test mode")
        test(args)


if __name__ == "__main__":
    main()
