# train.py

from grid_env import *
from episode_callback import EpisodeStatsCallback
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from Qlearning import QLearningAgent
from torch.utils.tensorboard import SummaryWriter
import time
from constants import *
from plot_metrics import plot_all_metrics

##==============================================================
## Cole's Experiments
##==============================================================
# different SB3 algorithms for training model
#-------------------------------------------
def train_PPO_model(grid_file: str, timesteps: int, model_name: str, log_name: str = None):
    if log_name is None:
        log_name = model_name

    vec_env = DummyVecEnv([lambda: GridWorldEnv(grid_file=grid_file)])
    
    log_path = os.path.join(LOGS["ppo"], log_name)
    model_save_path = os.path.join(MODELS["ppo"], model_name)

    model = PPO(
        "MlpPolicy",
        vec_env,
        ent_coef=0.5,
        gae_lambda=0.90,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1,
        tensorboard_log=log_path
    )

    callback = CustomTensorboardCallback()

    model.learn(total_timesteps=timesteps, callback=callback)
    
    model.save(model_save_path)
    print(f"\nPPO training complete. Model saved to {model_save_path} and logs to {log_path}")

    # generate graphs from csvs
    plots = plot_all_metrics(log_dir=log_path, output_dir=os.path.join(log_path, "plots"))
    
    print("\n=== Metrics Plots Generated ===")
    for csv_file, plot_list in plots.items():
        print(f"\n{csv_file}:")
        for p in plot_list:
            print(f"  {p}")
    
    return model

# SAC requires continuous action sapce

# training utils
#-------------------------------------------------
def load_model(model_path: str, env):
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model file not found at: {model_path}.zip")

    vec_env = DummyVecEnv([lambda: env])
    model = PPO.load(model_path, env=vec_env)
    return model

def evaluate_model(env, model, n_eval_episodes=5, sleep_time=0.1, render: bool = True, verbose: bool = False):
    total_rewards = []
    success_count = 0

    for ep in range(n_eval_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        terminated = False
        truncated = False
        episode_reward = 0

        print(f"\n--- Episode {ep + 1} ---")
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if verbose:
                print("reward", reward)
            if render:
                env.render_pygame()
                time.sleep(sleep_time)
            episode_reward += reward

        total_rewards.append(episode_reward)
        if terminated: # if agent reached exit
            success_count += 1
            
    mean_reward = sum(total_rewards) / len(total_rewards)
    print(f"\n Evaluation complete over {n_eval_episodes} episodes")
    print(f" Mean Reward: {mean_reward:.2f}")
    print(f" Successful Episodes (Reached Goal): {success_count} / {n_eval_episodes}")
    print("\n Final Episode Summary:")
    env.episode_summary()

def load_model_and_evaluate(model_filename: str, env, n_eval_episodes=20, sleep_time=0.1, render: bool = True, verbose: bool = True):
    """
    Load a model by filename and evaluate.
    """
    model_path = os.path.join(MODELS["ppo"], model_filename)
    model = load_model(model_path, env)
    evaluate_model(env, model, n_eval_episodes=n_eval_episodes, sleep_time=sleep_time, render=render, verbose=verbose)

def list_models():
    for f in os.listdir(MODELS["ppo"]):
        if f.endswith(".zip"):
            print(f)
