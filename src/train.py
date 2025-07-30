# train.py

import os
import time
import re
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO
from src.cnn_feature_extractor import GridCNNExtractor

from src.BatteryPredictorEnv import BatteryPredictorEnv
from src.constants import FIXED_GRID_DIR, SAVE_DIR

# ===================================================================
# --- Custom Callback for Logging ---
# ===================================================================

class PredictorTensorboardCallback(BaseCallback):
    """
    Logs MAE, RMSE, MAPE, and mean reward globally and for first 4 sensors to TensorBoard and console.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.all_predictions = []
        self.all_actuals = []
        self.all_rewards = []
        self.num_sensors = None

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        infos = self.locals.get('infos', [{}])
        if infos and isinstance(infos[0], dict):
            # Print or log from info (e.g., every 100 steps)
            if self.num_timesteps % 100 == 0:
                mae = infos[0].get('mae')
                rmse = infos[0].get('rmse')
                mape = infos[0].get('mape')
                mean_reward = infos[0].get('mean_reward')
                # Also log them to SB3 logger with standard prefix for console table
                if mae is not None:
                    self.logger.record("custom/MAE", mae)
                if rmse is not None:
                    self.logger.record("custom/RMSE", rmse)
                if mape is not None:
                    self.logger.record("custom/MAPE", mape)
                if mean_reward is not None:
                    self.logger.record("custom/mean_reward", mean_reward)
        return True

# ===================================================================
# --- Unified PPO-LSTM Training Function ---
# ===================================================================

def train_predictor_model(grid_file: str,
                          n_miners: int,
                          timesteps: int,
                          experiment_folder_name: str,
                          n_envs: int = 4,
                          is_cnn: bool = False):
    """
    Initializes and trains the RecurrentPPO battery predictor model, saving all
    artifacts into a structured experiment/run directory.

    Args:
        grid_file (str): The name of the grid file for the simulation.
        n_miners (int): The number of autonomous miners in the simulation.
        timesteps (int): The total number of training timesteps.
        experiment_folder_name (str): The unique name for this experiment.
        n_envs (int): The number of parallel environments to use for training.
    """
    if is_cnn and "is_cnn" not in experiment_folder_name:
        experiment_folder_name += "_cnn"
    
    # --- Step 1: Create the hierarchical directory structure ---
    base_log_path = os.path.join(SAVE_DIR, experiment_folder_name)
    os.makedirs(base_log_path, exist_ok=True)

    # Find the next available run number (e.g., PPO_1, PPO_2)
    run_num = 1
    existing_runs = [d for d in os.listdir(base_log_path) if d.startswith("RecurrentPPO_")]
    if existing_runs:
        max_run = max([int(r.split('_')[-1]) for r in existing_runs])
        run_num = max_run + 1
    
    # This is the final directory where everything for this run will be saved
    log_dir = os.path.join(base_log_path, f"RecurrentPPO_{run_num}")
    model_save_path = os.path.join(log_dir, "model.zip")
    os.makedirs(log_dir, exist_ok=True)

    # --- Step 2: Create the Vectorized Environment ---
    vec_env = make_vec_env(
        lambda: BatteryPredictorEnv(grid_file=grid_file, n_miners=n_miners),
        n_envs=n_envs
    )

    # --- Step 3: Define the RecurrentPPO Model ---
    # With sb3-contrib, you pass the policy name string 'MlpLstmPolicy'

    if is_cnn:
        policy_kwargs = dict(
            features_extractor_class=GridCNNExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,        # adjust as needed
                grid_file=grid_file,     # to let CNN arch auto-adapt
                backbone="seq"           # use seq by default
            )
        )
    else:
        policy_kwargs = {}

    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=1024,
        batch_size=64, # Note: batch_size for RecurrentPPO is per environment
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=log_dir
    )

    # --- Step 4: Train the Model ---
    print(f"--- Starting Training ---")
    print(f"All artifacts will be saved in: {log_dir}")
    
    callback = PredictorTensorboardCallback()
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar= True)
    print("--- Training Complete ---")

    # --- Step 5: Save the Final Model ---
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    vec_env.close()
    
    return log_dir # Return the path to the run folder for evaluation

def train_all_predictors(timesteps: int = 1_000_000):
    """
    Trains multiple RecurrentPPO predictor models using the current battery
    prediction setup. Each configuration defines the grid, number of miners, etc.
    """
    def attach_run_folder_names(model_configs):
        for config in model_configs:
            grid_name = os.path.splitext(config["grid_file"])[0]
            folder_parts = [grid_name, f"{config['n_miners']}miners"]
            if config.get("tag"):
                folder_parts.append(config["tag"])
            config["experiment_folder_name"] = "_".join(folder_parts)

    models_to_train = [
        {
            "grid_file": "mine_100x100.txt",
            "n_miners": 40,
            "is_cnn": True
        },
        {
            "grid_file": "mine_50x50.txt",
            "n_miners": 20,
            "is_cnn": True
        },
            {
            "grid_file": "mine_30x30.txt",
            "n_miners": 10,
            "is_cnn": True
        },
        {
            "grid_file": "mine_20x20.txt",
            "n_miners": 5,
            "is_cnn": True
        }    
    ]

    attach_run_folder_names(models_to_train)

    for config in models_to_train:
        print(f"\n===== Training Predictor: {config['experiment_folder_name']} =====")
        run_path = train_predictor_model(
            grid_file=config["grid_file"],
            n_miners=config["n_miners"],
            timesteps=timesteps,
            experiment_folder_name=config["experiment_folder_name"],
            n_envs=4
        )
        print(f"===== Finished: {run_path} =====")

# ===================================================================
# --- Evaluation Functions for the Predictor ---
# ===================================================================

def evaluate_predictor(run_path: str, grid_file: str, n_miners: int, eval_steps: int = 500):
    """
    Loads and evaluates a single trained predictor model from its run folder.
    """
    model_path = os.path.join(run_path, "model.zip")
    print(f"\n--- Evaluating Predictor Model: {os.path.basename(run_path)} from experiment {os.path.basename(os.path.dirname(run_path))} ---")
    if not os.path.exists(model_path):
        print(f"Error: model.zip not found in {run_path}")
        return

    eval_env = BatteryPredictorEnv(grid_file=grid_file, n_miners=n_miners)
    # Use RecurrentPPO to load the model
    model = RecurrentPPO.load(model_path, env=eval_env)

    all_predictions, all_actuals = [], []
    obs, _ = eval_env.reset()
    # For RecurrentPPO, we need to manage the LSTM state and done signals
    lstm_states = None
    dones = np.zeros((1,)) # Start with a single environment not being done
    
    for step in range(eval_steps):
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=dones, deterministic=True
        )
        obs, reward, terminated, truncated, info = eval_env.step(action)
        dones[0] = terminated or truncated
        
        predicted_batteries = action * 100.0
        actual_batteries = np.array(list(eval_env.simulator.sensor_batteries.values()))
        all_predictions.append(predicted_batteries)
        all_actuals.append(actual_batteries)
        
        if dones[0]:
            obs, _ = eval_env.reset()
            # The LSTM state is reset automatically when episode_start is True
    eval_env.close()

    predictions_arr, actuals_arr = np.array(all_predictions), np.array(all_actuals)
    mae = np.mean(np.abs(predictions_arr - actuals_arr))
    rmse = np.sqrt(np.mean((predictions_arr - actuals_arr)**2))
    epsilon = 1e-12
    mape = np.mean(np.abs((actuals_arr - predictions_arr) / (actuals_arr + epsilon))) * 100

    print(f"  -> Mean Absolute Error (MAE): {mae:.3f}%")
    print(f"  -> Root Mean Squared Error (RMSE): {rmse:.3f}%")
    print(f"  -> Mean Absolute Percentage Error (MAPE): {mape:.3f}%")

    # Plotting Results and saving them to the run folder
    plt.figure(figsize=(15, 8))
    for i in range(min(4, eval_env.num_sensors)):
        plt.subplot(2, 2, i + 1)
        plt.plot(actuals_arr[:, i], label='Actual Battery', color='blue')
        plt.plot(predictions_arr[:, i], label='Predicted Battery', color='red', linestyle='--')
        plt.title(f'Sensor #{i+1} Prediction vs. Actual')
        plt.xlabel('Timestep'); plt.ylabel('Battery %')
        plt.legend(); plt.grid(True)
    plt.tight_layout()
    
    # Create a 'plots' subdirectory inside the run folder
    plot_dir = os.path.join(run_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_save_path = os.path.join(plot_dir, "evaluation_plot.png")
    
    plt.savefig(plot_save_path)
    print(f"  -> Evaluation plot saved to: {plot_save_path}")
    plt.close()

def evaluate_all_predictors(base_dir="saved_experiments/predictor_experiments", eval_steps=500):
    """
    Finds and evaluates all saved predictor models, parsing config from their
    experiment folder names (e.g., 'mine_30x30_15miners_baseline').
    """
    print(f"\n--- Evaluating All Predictor Models in: {base_dir} ---")
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    for experiment_name in os.listdir(base_dir):
        experiment_path = os.path.join(base_dir, experiment_name)
        if not os.path.isdir(experiment_path):
            continue
        
        for run_name in os.listdir(experiment_path):
            run_path = os.path.join(experiment_path, run_name)
            if not os.path.isdir(run_path) or not os.path.exists(os.path.join(run_path, "model.zip")):
                continue

            try:
                parts = experiment_name.split('_')
                grid_file = f"{parts[0]}_{parts[1]}.txt"
                miners_part = [p for p in parts if "miners" in p][0]
                n_miners = int(re.search(r'(\d+)miners', miners_part).group(1))
                
                evaluate_single_predictor(
                    run_path=run_path,
                    grid_file=grid_file,
                    n_miners=n_miners,
                    eval_steps=eval_steps,
                )
                
            except Exception as e:
                print(f"Could not parse or evaluate run '{run_name}' in experiment '{experiment_name}'. Error: {e}")
                continue
