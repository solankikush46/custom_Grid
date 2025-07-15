# train.py
from grid_env import *
from episode_callback import EpisodeStatsCallback
import os
import numpy as np
import gym
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from Qlearning import QLearningAgent
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
from constants import *
from plot_metrics import *
from cnn_feature_extractor import CustomGridCNNWrapper, GridCNNExtractor

##==============================================================
## Unified PPO Training Function
##==============================================================
def train_PPO_model(grid_file: str,
                    timesteps: int,
                    folder_name: str,
                    reset_kwargs: dict = {},
                    is_cnn: bool = False,
                    features_dim: int = 128,
                    battery_truncation=False):
    
    # Initialize environment (wrapped if CNN)
    env = GridWorldEnv(grid_file=grid_file, is_cnn=is_cnn, reset_kwargs=reset_kwargs, battery_truncation=battery_truncation)
    if is_cnn:
        env = CustomGridCNNWrapper(env)

    vec_env = DummyVecEnv([lambda: env])

    base_log_path = os.path.join(SAVE_DIR, folder_name)

    # Setup CNN policy kwargs if needed
    policy_kwargs = None
    if is_cnn:
        policy_kwargs = {
            "features_extractor_class": GridCNNExtractor,
            "features_extractor_kwargs": {"features_dim": features_dim},
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        }

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        ent_coef=0.1, #0.5,
        gae_lambda=0.90,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        clip_range_vf=0.5,
        tensorboard_log=base_log_path,
        verbose=1,
        policy_kwargs=policy_kwargs
    )

    callback = CustomTensorboardCallback()

    model.learn(total_timesteps=timesteps, callback=callback)

    log_dir = get_latest_run_dir(base_log_path)
    model_save_path = os.path.join(log_dir, "model.zip")
    
    model.save(model_save_path)
    print("\nPPO training complete. Logs and model stored to %s" % log_dir)

    # generate graphs from csvs using chunked smoothing
    grid_area = env.n_rows * env.n_cols
    num_points = int(max(20, grid_area // 10))
    plots = plot_all_metrics(log_dir=log_dir, num_points=num_points)

    print("\n=== Metrics Plots Generated ===")
    for csv_file, plot_list in plots.items():
        print(f"\n{csv_file}:")
        for p in plot_list:
            print(f"  {p}")

    return model

# training utils
#-------------------------------------------------
def load_model(experiment_folder: str, grid_file: str, is_cnn: bool = False, reset_kwargs: dict = {}):
    """
    Load a PPO model with environment matching the training setup.
    Args:
        experiment_folder: full path to the experiment folder containing model.zip and logs
        grid_file: grid text filename used for environment construction
        is_cnn: whether to wrap env for CNN input
        reset_kwargs: optional reset keyword args (e.g., battery overrides)
    """
    print("experiment_folder", experiment_folder)
    model_path = os.path.join(experiment_folder, "model")
    print("model_path", model_path)

    env = GridWorldEnv(grid_file=grid_file, is_cnn=is_cnn, reset_kwargs=reset_kwargs)
    if is_cnn:
        env = CustomGridCNNWrapper(env)

    print("Loading env observation space:", env.observation_space)
    vec_env = DummyVecEnv([lambda: env])

    model = PPO.load(model_path, env=vec_env)
    return model

def evaluate_model(env, model, n_eval_episodes=20, sleep_time=0.1, render: bool = True, verbose: bool = True,
                   halfsplit=False):
    """
    Evaluate the model in the given environment for a number of episodes,
    printing agent's position, reward, and action at every timestep, and summarizing performance.
    """
    total_rewards = []
    total_steps = 0
    success_count = 0
    total_collisions = 0

    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
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
            reached_exit = env.agent_reached_exit()
            success_count += int(reached_exit)

            if verbose:
                action_dir = ACTION_NAMES.get(int(action), f"Unknown({action})")
                print(f"Episode {ep + 1} Step {step_num}: Pos={agent_pos}, Action={action} ({action_dir}), Reward={reward:.2f}")
                for name, val in subrewards.items():
                    print(f"  └─ {name}: {val:.2f}")

            if render:
                env.render_pygame()
                time.sleep(sleep_time)

            step_num += 1

        total_collisions += info.get("obstacle_hits", 0)
        total_steps += step_num
        total_rewards.append(ep_reward)

        if verbose:
            print(f"Episode {ep + 1} complete: Total Reward = {ep_reward:.2f}")

    mean_reward = sum(total_rewards) / n_eval_episodes
    success_rate = success_count / n_eval_episodes
    mean_col = total_collisions / n_eval_episodes
    avg_steps = total_steps / n_eval_episodes

    print("\n=== Evaluation Summary ===")
    print(f"Total Episodes: {n_eval_episodes}")
    print(f"Reached Exit: {success_count}/{n_eval_episodes} ({success_rate:.1%})")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Mean Obstacle Hits: {mean_col:.2f}")
    print(f"Average Steps per Episode: {avg_steps:.1f}")

def load_model_and_evaluate(model_folder: str, grid_file: str, is_cnn: bool = False, reset_kwargs: dict = {},
                            n_eval_episodes=20, sleep_time=0.1, render: bool = True, verbose: bool = True):
    """
    Load a model by experiment folder and evaluate it in the matching environment.
    """
    model = load_model(model_folder, grid_file, is_cnn, reset_kwargs)
    evaluate_model(env=model.get_env().envs[0], model=model, n_eval_episodes=n_eval_episodes,
                   sleep_time=sleep_time, render=render, verbose=verbose)

def get_halfsplit_battery_overrides(grid_path: str) -> dict:
    """
    Returns a battery override dictionary where:
    - Top half of sensors have 100.0 battery
    - Bottom half of sensors have 0.0 battery
    """
    sensor_positions = []
    with open(grid_path, "r") as f:
        lines = [line.strip() for line in f]
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == SENSOR:
                    sensor_positions.append((r, c))

    if not sensor_positions:
        raise ValueError(f"No sensors found in: {grid_path}")

    n_rows = len(lines)
    mid_row = n_rows // 2

    battery_overrides = {
        (r, c): 100.0 if r < mid_row else 0.0
        for r, c in sensor_positions
    }

    return battery_overrides

def train_quick_junk_model(grid_file: str, is_cnn: bool = False):
    '''
    Very small training for quick testing
    '''
    junk_folder_name = grid_file + "__" + "junk"
    if is_cnn:
        junk_folder_name += "_cnn"
    timesteps = 500 
    
    model = train_PPO_model(
        grid_file=grid_file,
        timesteps=timesteps,
        folder_name=junk_folder_name,
        is_cnn=is_cnn
    )
    
    print(f"Trained junk model '{junk_folder_name}' for {timesteps} timesteps.")
    return junk_folder_name
