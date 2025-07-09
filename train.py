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
                if char == SENSOR:
                    sensor_positions.append((r, c))

    if not sensor_positions:
        raise ValueError("No sensors found in grid file.")

    # Determine halfway row (based on max row)
    max_row = max(pos[0] for pos in sensor_positions)
    halfway_row = max_row // 2

    battery_overrides = {
        pos: 100.0 if pos[0] <= halfway_row else 0.0
        for pos in sensor_positions
    }

    print("Battery override applied (top = 100, bottom = 0):")
    for pos, val in battery_overrides.items():
        print(f"  {pos}: {val}")

    # Step 2: Create env with override
    env = GridWorldEnv(grid_file=grid_file, battery_overrides=battery_overrides)
    obs, _ = env.reset()

    wrapped_env = CustomGridCNNWrapper(env)
    vec_env = make_vec_env(lambda: wrapped_env, n_envs=1)

    # Step 3: Define CNN-based policy
    policy_kwargs = {
        "features_extractor_class": GridCNNExtractor,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "net_arch": dict(pi=[64, 64], vf=[64, 64])
    }

    # Step 4: Train model
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="cuda"
    )

    chunk_steps = total_timesteps // 10
    for i in range(1, 11):
        print(f"\nTraining chunk {i}/10: {chunk_steps} steps...")
        model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)

        # Save checkpoint
        checkpoint_path = f"{save_path}_{i * 10}pct"
        model.save(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    return model
