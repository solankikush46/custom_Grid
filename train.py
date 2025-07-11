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
from constants import *
from plot_metrics import plot_all_metrics
from cnn_feature_extractor import CustomGridCNNWrapper, GridCNNExtractor

##==============================================================
## Unified PPO Training Function (MLP or CNN)
##==============================================================
def train_PPO_model(grid_file: str,
                    timesteps: int,
                    model_name: str,
                    log_name: str = None,
                    reset_kwargs: dict = {},
                    is_cnn: bool = False,
                    features_dim: int = 128):
    if log_name is None:
        log_name = model_name

    # Initialize environment (wrapped if CNN)
    env = GridWorldEnv(grid_file=grid_file, is_cnn=is_cnn, reset_kwargs=reset_kwargs)
    if is_cnn:
        env = CustomGridCNNWrapper(env)

    vec_env = DummyVecEnv([lambda: env])

    log_path = os.path.join(LOGS["ppo"], log_name)
    model_save_path = os.path.join(MODELS["ppo"], model_name)

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
        ent_coef=0.5,
        gae_lambda=0.90,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        #gamma=0.99,
        #clip_range=0.2,
        #clip_range_vf=0.5,
        tensorboard_log=log_path,
        verbose=1,
        policy_kwargs=policy_kwargs
    )
    '''
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        ent_coef=0.5,
        gae_lambda=0.90,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1
    )
    '''

    callback = CustomTensorboardCallback()

    model.learn(total_timesteps=timesteps, callback=callback)

    model.save(model_save_path)
    print(f"\nPPO training complete. Model saved to {model_save_path} and logs to {log_path}")

    # generate graphs from csvs using chunked smoothing
    grid_area = env.n_rows * env.n_cols
    num_points = int(max(20, grid_area // 10))
    plots = plot_all_metrics(log_dir=log_path, num_points=num_points)

    print("\n=== Metrics Plots Generated ===")
    for csv_file, plot_list in plots.items():
        print(f"\n{csv_file}:")
        for p in plot_list:
            print(f"  {p}")

    return model

# training utils
#-------------------------------------------------
def load_model(model_path: str, grid_file: str, is_cnn: bool = False, reset_kwargs: dict = {}):
    """
    Load a PPO model with environment matching the training setup.
    Args:
        model_path: full path to the saved model (without .zip)
        grid_file: grid text filename used for environment construction
        is_cnn: whether to wrap env for CNN input
        reset_kwargs: optional reset keyword args (e.g., battery overrides)
    """
    env = GridWorldEnv(grid_file=grid_file, is_cnn=is_cnn, reset_kwargs=reset_kwargs)
    if is_cnn:
        env = CustomGridCNNWrapper(env)

    print("Loading env observation space:", env.observation_space)

    vec_env = DummyVecEnv([lambda: env])

    model = PPO.load(model_path, env=vec_env)
    return model

def evaluate_model(env, model, n_eval_episodes=20, sleep_time=0.1, render: bool = True, verbose: bool = True):
    """
    Evaluate the model in the given environment for a number of episodes,
    printing agent's position, reward, and action at every timestep, and summarizing performance.
    """
    total_rewards = []
    total_steps = 0
    success_count = 0  # Track number of episodes where agent reached the goal

    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        step_num = 0
        reached_exit = False

        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            agent_pos = info.get('agent_pos', None)
            subrewards = info.get('subrewards', {})

            reached_exit = bool(info.get("reward") == 400)

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

        total_steps += step_num
        total_rewards.append(ep_reward)
        if reached_exit:
            success_count += 1

        if verbose:
            print(f"Episode {ep + 1} complete: Total Reward = {ep_reward:.2f}")

    mean_reward = sum(total_rewards) / n_eval_episodes
    success_rate = success_count / n_eval_episodes
    avg_steps = total_steps / n_eval_episodes

    print("\n=== Evaluation Summary ===")
    print(f"Total Episodes: {n_eval_episodes}")
    print(f"Reached Exit: {success_count}/{n_eval_episodes} ({success_rate:.1%})")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Average Steps per Episode: {avg_steps:.1f}")

def load_model_and_evaluate(model_filename: str, grid_file: str, is_cnn: bool = False, reset_kwargs: dict = {},
                            n_eval_episodes=20, sleep_time=0.1, render: bool = True, verbose: bool = True):
    """
    Load a model by filename and evaluate it in the matching environment.
    """
    model_path = os.path.join(MODELS["ppo"], model_filename)
    model = load_model(model_path, grid_file, is_cnn, reset_kwargs)
    evaluate_model(env=model.get_env().envs[0], model=model, n_eval_episodes=n_eval_episodes,
                   sleep_time=sleep_time, render=render, verbose=verbose)

def list_models():
    for f in os.listdir(MODELS["ppo"]):
        if f.endswith(".zip"):
            print(f)

##==============================================================
## Utilities for Half-Split Battery Scenarios
##==============================================================

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

def train_halfsplit_model(grid_filename: str, timesteps: int, battery_overrides: dict, is_cnn: bool = False, model_name=None):
    """
    Trains a PPO model using the half-split battery override.
    """
    if not model_name:
        model_name = f"{'cnn_' if is_cnn else ''}battery_halfsplit_{grid_filename.replace('.txt','')}"
    reset_kwargs = {"battery_overrides": battery_overrides}

    model = train_PPO_model(
        grid_file=grid_filename,
        timesteps=timesteps,
        model_name=model_name,
        reset_kwargs=reset_kwargs,
        is_cnn=is_cnn
    )

    return model_name, model

def evaluate_halfsplit_model(model_name: str, grid_filename: str, battery_overrides: dict,
                             episodes: int = 3, render: bool = True, verbose: bool = True):
    """
    Evaluates a trained PPO model on a half-split battery scenario.
    Battery overrides must be passed explicitly.
    """
    reset_kwargs = {"battery_overrides": battery_overrides}
    env = GridWorldEnv(grid_file=grid_filename, reset_kwargs=reset_kwargs)

    if verbose:
        print(f"Evaluating model '{model_name}' on grid '{grid_filename}' with battery overrides:")
        for sensor_pos, battery_level in battery_overrides.items():
            print(f"  Sensor at {sensor_pos}: battery = {battery_level}")

    load_model_and_evaluate(
        model_filename=model_name,
        grid_file=grid_filename,
        reset_kwargs=reset_kwargs,
        n_eval_episodes=episodes,
        sleep_time=0.1,
        render=render,
        verbose=verbose
    )

def train_quick_junk_model(grid_file: str, is_cnn: bool = False):
    # Very small training for quick testing
    junk_model_name = f"junk_{'cnn_' if is_cnn else ''}{grid_file.replace('.txt','')}"
    timesteps = 500  # super low for quick test
    
    model = train_PPO_model(
        grid_file=grid_file,
        timesteps=timesteps,
        model_name=junk_model_name,
        is_cnn=is_cnn
    )
    
    print(f"Trained junk model '{junk_model_name}' for {timesteps} timesteps.")
    return junk_model_name
