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

    env = GridWorldEnv(grid_file=grid_file)
    vec_env = DummyVecEnv([lambda: env])
    
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

    # generate graphs from csvs, with rolling window being
    # 5% of grid area
    grid_area = env.n_rows * env.n_cols
    rolling_window = int(max(1, 0.05 * grid_area))
    plots = plot_all_metrics(log_dir=log_path, rolling_window=rolling_window)
    
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

def evaluate_model(env, model, n_eval_episodes=20, sleep_time=0.1, render: bool = True, verbose: bool = True, reset_kwargs=None):
    """
    Evaluate the model in the given environment for a number of episodes,
    printing agent's position, reward, and action at every timestep.
    """
    total_rewards = []

    for ep in range(n_eval_episodes):
        obs, _ = env.reset(**(reset_kwargs or {}))
        done = False
        ep_reward = 0.0
        step_num = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            agent_pos = info.get('agent_pos', None)
            subrewards = info.get('subrewards', {})
            
            if verbose:
                action_dir = ACTION_NAMES.get(int(action), f"Unknown({action})")
                print(f"Episode {ep + 1} Step {step_num}: Pos={agent_pos}, Action={action} ({action_dir}), Reward={reward:.2f}")
                if subrewards:
                    for name, val in subrewards.items():
                        print(f"  └─ {name}: {val:.2f}")

            if render:
                env.render_pygame()
                time.sleep(sleep_time)

            step_num += 1

        total_rewards.append(ep_reward)
        if verbose:
            print(f"Episode {ep + 1} complete: Total Reward = {ep_reward:.2f}")

    mean_reward = sum(total_rewards) / n_eval_episodes
    print(f"\nEvaluation complete: Mean reward = {mean_reward:.2f}")

def load_model_and_evaluate(model_filename: str, env, n_eval_episodes=20, sleep_time=0.1, render: bool = True, verbose: bool = True, reset_kwargs=None):
    """
    Load a model by filename and evaluate.
    """
    model_path = os.path.join(MODELS["ppo"], model_filename)
    model = load_model(model_path, env)
    evaluate_model(env, model, n_eval_episodes=n_eval_episodes, sleep_time=sleep_time, render=render, verbose=verbose, reset_kwargs=reset_kwargs)

def list_models():
    for f in os.listdir(MODELS["ppo"]):
        if f.endswith(".zip"):
            print(f)
