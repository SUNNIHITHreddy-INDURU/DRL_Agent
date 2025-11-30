import argparse
import os

import gymnasium as gym
import highway_env
import numpy as np

from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# QR-DQN from sb3-contrib
from sb3_contrib import QRDQN

# Import custom env so registration runs
import multi_stage_env


# ---------------------------------------------------------
# Helper: create environment with unwrapping + observation config
# ---------------------------------------------------------
def make_env(env_name, obs_type="lidar", render_mode=None, seed=0):
    """Creates an environment, unwraps Gymnasium wrappers, and applies observation config."""
    def _init():
        env = gym.make(env_name, render_mode=render_mode)

        # Unwrap Gymnasium wrappers (OrderEnforcing, PassiveEnvChecker, etc.)
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        # CONFIGURE OBSERVATION
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

        env.reset(seed=seed)
        return env

    return _init


# ---------------------------------------------------------
# Experiment ID → environment name + observation type
# ---------------------------------------------------------
def get_env_name_from_id(exp_id):
    env_map = {
        # Task 1: Highway
        1: ("highway-v0", "lidar"),
        2: ("highway-v0", "lidar"),
        3: ("highway-v0", "grayscale"),
        4: ("highway-v0", "grayscale"),

        # Task 1: Merge
        5: ("merge-v0", "lidar"),
        6: ("merge-v0", "lidar"),
        7: ("merge-v0", "grayscale"),
        8: ("merge-v0", "grayscale"),

        # Task 1: Intersection
        9: ("intersection-v0", "lidar"),
        10: ("intersection-v0", "lidar"),
        11: ("intersection-v0", "grayscale"),
        12: ("intersection-v0", "grayscale"),

        # 🔥 FIXED: Your improved custom environment
        13: ("highway-construction-v0", "lidar"),
        14: ("highway-construction-v0", "lidar"),

        # Other team’s env
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
    print("\n" + "=" * 80)
    print(f"TRAINING | Algo = {args.algo.upper()} | Exp-ID = {args.exp_id}")
    print(f"Env     : {args.env_name}")
    print(f"ObsType : {args.obs_type}")
    print(f"Seed    : {args.seed}")
    print(f"Steps   : {args.total_timesteps}")
    print("=" * 80 + "\n")

    log_dir = f"logs/exp_{args.exp_id}_{args.algo}_{args.obs_type}"
    model_dir = f"models/exp_{args.exp_id}"
    tb_dir = f"tensorboard/exp_{args.exp_id}_{args.algo}_{args.obs_type}"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # -----------------------------------------------------
    # SubprocVecEnv (4 environments in parallel)
    # -----------------------------------------------------
    env_fns = [
        make_env(args.env_name, args.obs_type, seed=args.seed + i)
        for i in range(4)
    ]
    env = SubprocVecEnv(env_fns)

    # Eval environment → single environment (DummyVecEnv)
    eval_env = DummyVecEnv([make_env(args.env_name, args.obs_type, seed=args.seed)])

    # Select algorithm
    algo_class = {
        "ppo": PPO,
        "dqn": DQN,
        "a2c": A2C,
        "sac": SAC,
        "qrdqn": QRDQN,
    }[args.algo.lower()]

    # -----------------------------------------------------
    # Hyperparameters (we keep your optimized QR-DQN config)
    # -----------------------------------------------------
    if args.algo.lower() == "qrdqn":
        model_kwargs = {
            "learning_rate": 3e-4,
            "buffer_size": 200_000,
            "learning_starts": 5_000,
            "batch_size": 128,
            "gamma": 0.995,
            "train_freq": (4, "step"),
            "gradient_steps": 2,
            "target_update_interval": 2_000,
            "exploration_fraction": 0.25,
            "exploration_final_eps": 0.02,
            "policy_kwargs": dict(
                net_arch=[256, 256],
                n_quantiles=50,
            ),
            "tensorboard_log": tb_dir,
            "verbose": 1,
        }

    elif args.algo.lower() == "ppo":
        model_kwargs = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "tensorboard_log": tb_dir,
            "verbose": 1,
        }

    else:
        model_kwargs = {"tensorboard_log": tb_dir, "verbose": 1}

    # Create model with seed
    model = algo_class(
        "MlpPolicy",
        env,
        seed=args.seed,
        **model_kwargs
    )

    # Checkpoints + evaluations
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

    # Start training
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = f"{model_dir}/final_model"
    model.save(final_path)
    print(f"\nModel saved at: {final_path}\n")

    env.close()
    eval_env.close()


# ---------------------------------------------------------
# TEST FUNCTION
# ---------------------------------------------------------
def test(args):
    print("\n" + "=" * 80)
    print(f"TESTING | Algo = {args.algo.upper()} | Exp-ID = {args.exp_id}")
    print(f"Env     : {args.env_name}")
    print(f"ObsType : {args.obs_type}")
    print(f"Seed    : {args.seed}")
    print(f"Model   : {args.load_model}")
    print("=" * 80 + "\n")

    env = make_env(args.env_name, args.obs_type, render_mode="human", seed=args.seed)()

    algo_class = {
        "ppo": PPO,
        "dqn": DQN,
        "a2c": A2C,
        "sac": SAC,
        "qrdqn": QRDQN,
    }[args.algo.lower()]

    model = algo_class.load(args.load_model)

    rewards, lengths = [], []

    for ep in range(args.test_episodes):
        obs, _ = env.reset(seed=args.seed)
        done = False
        ep_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += r
            steps += 1
            if args.render:
                env.render()

        rewards.append(ep_reward)
        lengths.append(steps)
        print(f"Episode {ep+1}/{args.test_episodes}: Reward = {ep_reward:.2f}, Length = {steps}")

    print("\n" + "-" * 60)
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean Length: {np.mean(lengths):.2f}")
    print("-" * 60 + "\n")

    # Save results
    results_dir = f"results/exp_{args.exp_id}"
    os.makedirs(results_dir, exist_ok=True)
    np.savez(f"{results_dir}/test_results_exp_{args.exp_id}.npz",
             rewards=np.array(rewards),
             lengths=np.array(lengths),
             mean_reward=np.mean(rewards),
             std_reward=np.std(rewards))

    env.close()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DRL Training & Testing Pipeline")

    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--exp-id", type=int, required=True)
    parser.add_argument("--algo", type=str, default="qrdqn",
                        choices=["ppo", "dqn", "a2c", "sac", "qrdqn"])
    parser.add_argument("--env-name", type=str, default=None)
    parser.add_argument("--obs-type", type=str, default=None,
                        choices=["lidar", "grayscale"])
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--test-episodes", type=int, default=500)
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--render", action="store_true")

    # ⭐ NEW: Seed argument
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # Auto-detect env/obs from exp_id
    if args.env_name is None or args.obs_type is None:
        args.env_name, args.obs_type = get_env_name_from_id(args.exp_id)

    if args.mode == "train":
        train(args)
    else:
        if not args.load_model:
            raise ValueError("Must provide --load-model for test mode")
        test(args)


if __name__ == "__main__":
    main()
