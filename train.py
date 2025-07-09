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
## Cole's Experiments
##==============================================================
# different SB3 algorithms for training model
#-------------------------------------------
def train_PPO_model(grid_file: str, timesteps: int, model_name: str,
                    log_name: str = None, reset_kwargs: dict = None):
    if log_name is None:
        log_name = model_name

    env = GridWorldEnv(grid_file=grid_file)

    if reset_kwargs:
        env.reset(**reset_kwargs)
    else:
        env.reset()

    vec_env = DummyVecEnv([lambda: env])
    
    log_path = os.path.join(LOGS["ppo"], log_name)
    model_save_path = os.path.join(MODELS["ppo"], model_name)

    model = PPO(
        policy         = "MlpPolicy",
        env            = vec_env,
        learning_rate  = 3e-5,
        n_steps        = 2048,
        batch_size     = 2048,
        n_epochs       = 10,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        clip_range_vf  = 0.5,
        ent_coef       = 0.01,
        tensorboard_log= log_path,
        verbose        = 1,
        device         = "cuda"
    )

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
def load_model(model_path: str, env):
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model file not found at: {model_path}.zip")

    vec_env = DummyVecEnv([lambda: env])
    model = PPO.load(model_path, env=vec_env)
    return model

def evaluate_model(env, model, n_eval_episodes=20, sleep_time=0.1, render: bool = True, verbose: bool = True, reset_kwargs=None):
    """
    Evaluate the model in the given environment for a number of episodes,
    printing agent's position, reward, and action at every timestep, and summarizing performance.
    """
    total_rewards = []
    total_steps = 0
    success_count = 0  # Track number of episodes where agent reached the goal

    for ep in range(n_eval_episodes):
        obs, _ = env.reset(**(reset_kwargs or {}))
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

            # Check for exit condition — adjust if your env tracks it differently
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

##==============================================================
## Kush's Experiments
##==============================================================

def create_and_train_cnn_ppo_model(grid_file: str, total_timesteps: int = 100_000, save_path: str = "ppo_model", features_dim: int = 128) -> PPO:
    """
    Initializes PPO with a CNN feature extractor and applies a half-split battery override,
    where the top half of sensors get 100.0 and bottom half get 0.0.

    Args:
        grid_file (str): Grid layout filename.
        total_timesteps (int): Total number of timesteps to train.
        save_path (str): Where to save model checkpoints.
        features_dim (int): Output size of CNN feature extractor.

    Returns:
        PPO: Trained PPO model.
    """
    # Step 1: Determine battery override
    sensor_positions = []
    grid_path = os.path.join(FIXED_GRID_DIR, grid_file)
    
    with open(grid_path, "r") as f:
        for r, line in enumerate(f):
            for c, char in enumerate(line.strip()):
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

def train_halfsplit_model(grid_filename: str, timesteps: int, battery_overrides: dict):
    """
    Trains a PPO model using the half-split battery override.
    """
    model_name = f"battery_halfsplit_{grid_filename.replace('.txt','')}"
    reset_kwargs = {"battery_overrides": battery_overrides}

    model = train_PPO_model(
        grid_file=grid_filename,
        timesteps=timesteps,
        model_name=model_name,
        reset_kwargs=reset_kwargs
    )

    return model_name, model

def evaluate_halfsplit_model(model_name: str, grid_filename: str, battery_overrides: dict,
                             episodes: int = 3, render: bool = True, verbose: bool = True):
    """
    Evaluates a trained PPO model on a half-split battery scenario.
    Battery overrides must be passed explicitly.
    """
    env = GridWorldEnv(grid_file=grid_filename)
    reset_kwargs = {"battery_overrides": battery_overrides}

    if verbose:
        print(f"Evaluating model '{model_name}' on grid '{grid_filename}' with battery overrides:")
        for sensor_pos, battery_level in battery_overrides.items():
            print(f"  Sensor at {sensor_pos}: battery = {battery_level}")

    load_model_and_evaluate(
        model_filename=model_name,
        env=env,
        n_eval_episodes=episodes,
        sleep_time=0.1,
        render=render,
        verbose=verbose,
        reset_kwargs=reset_kwargs
    )
