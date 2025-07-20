# test.py

import time
import os
from grid_env import GridWorldEnv
from constants import *
import train
import grid_gen
import matplotlib.pyplot as plt
from cnn_feature_extractor import CustomGridCNNWrapper, GridCNNExtractor
from plot_metrics import *
import re

def test_manual_control(grid_file: str = "mine_20x20.txt"):
    """
    Launch manual control mode for the specified grid file.
    """
    env = GridWorldEnv(grid_file=grid_file)
    env.manual_control_loop()  

def generate_grid(rows: int, cols: int, obstacle_percentage: float,
                  n_sensors: int=0, filename: str = None
                  ) -> str:
    '''
    Generates and saves a grid file, returning the filename
    '''
    if filename is None:
        filename = f"grid_{rows}x{cols}_{int(obstacle_percentage * 100)}p.txt"
    save_path = os.path.join(FIXED_GRID_DIR, filename)

    grid_gen.gen_and_save_grid(
        rows=rows,
        cols=cols,
        obstacle_percentage=obstacle_percentage,
        n_sensors=n_sensors,
        place_agent=False,
        save_path=save_path
    )

    return filename

def simulate_battery_depletion(grid_file="mine_20x20.txt", max_steps=100_000):
    """
    Simulates environment until all sensor batteries deplete or max_steps is reached.
    Saves and plots battery levels over time for each sensor.
    """
    env = GridWorldEnv(grid_file)
    obs, _ = env.reset()
    step = 0
    battery_history = {pos: [] for pos in env.sensor_batteries}
    all_empty = False

    while step < max_steps and not all_empty:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        for pos in battery_history:
            battery_history[pos].append(info["sensor_batteries"].get(pos, 0.0))

        all_empty = all(v <= 0.0 for v in info["sensor_batteries"].values())
        step += 1

    # Plot battery levels
    plt.figure(figsize=(10, 6))
    for pos, levels in battery_history.items():
        label = f"Sensor {pos}"
        plt.plot(range(len(levels)), levels, label=label)

    plt.xlabel("Timestep")
    plt.ylabel("Battery Level (%)")
    plt.title("Sensor Battery Depletion Over Time")
    plt.legend(loc='lower left', fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def test_observation_space():
    env = GridWorldEnv(grid_height=10, grid_width=10, obstacle_percentage=0.2, n_sensors=5)
    obs, _ = env.reset()

    print("\nObservation Length:", len(obs))
    print("Grid Dimensions:", env.n_rows, "x", env.n_cols)
    print("Sample Observation (first 100 values):")
    print(np.array(obs[:100]))

    # Count how many of each value type (just for sanity)
    unique, counts = np.unique(obs, return_counts=True)
    print("\nUnique values in observation and their counts:")
    for val, cnt in zip(unique, counts):
        print(f"Value {val:.2f}: {cnt} times")

def best_matching_grid(experiment_name: str, grid_dir: str) -> str:
    grid_name = experiment_name.split("__")[0] + ".txt"
    print(grid_name)
    grid_file_path = os.path.join(grid_dir, grid_name)
    if not os.path.exists(grid_file_path):
        raise FileNotFoundError(f"Grid file not found: {grid_file_path}")
    return grid_name

def evaluate_ppo_run(ppo_path, experiment_name, n_eval_episodes, render, verbose):
    inferred_grid = best_matching_grid(experiment_name, FIXED_GRID_DIR)
    grid_file_path = os.path.join(FIXED_GRID_DIR, inferred_grid)

    is_cnn = "cnn" in experiment_name.lower()
    is_halfsplit = "halfsplit" in experiment_name.lower()

    print(f"\nEvaluating {ppo_path} using grid {inferred_grid} (CNN={is_cnn}, Halfsplit={is_halfsplit})...")

    reset_kwargs = {}
    if is_halfsplit:
        battery_overrides = get_halfsplit_battery_overrides(grid_file_path)
        reset_kwargs["battery_overrides"] = battery_overrides

    model = train.load_model(ppo_path, grid_file=inferred_grid, is_cnn=is_cnn, reset_kwargs=reset_kwargs)
    env = model.get_env().envs[0]

    train.evaluate_model(env, model, n_eval_episodes=n_eval_episodes, render=render, halfsplit=is_halfsplit, verbose=verbose)

def evaluate_all_models(base_dir=SAVE_DIR, n_eval_episodes=10, render=True, verbose=True, dos=[]):
    """
    Evaluates all PPO models under each experiment in `base_dir`.
    """
    def extract_ppo_number(name):
        try:
            return int(name.split('_')[-1])
        except ValueError:
            return -1

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory {base_dir} not found")

    for experiment_name in os.listdir(base_dir):
        experiment_path = os.path.join(base_dir, experiment_name)
        if not os.path.isdir(experiment_path):
            continue

        print("Processing", experiment_name)

        for ppo_run in sorted(os.listdir(experiment_path), key=extract_ppo_number, reverse=True):
            ppo_path = os.path.join(experiment_path, ppo_run)
            model_path = os.path.join(ppo_path, "model.zip")

            if not os.path.isfile(model_path):
                continue

            flag = False
            for do in dos:
                if not do in ppo_path:
                    print("ignoring", ppo_path)
                    flag = True
                    break
            if flag or "sigmoid" in ppo_path:
                continue

            evaluate_ppo_run(ppo_path, experiment_name, n_eval_episodes, render, verbose)

def train_all_models(timesteps: int = 1_000_000):
    """
    Trains PPO models with support for halfsplit battery overrides when
    specified
    """
    models_to_train = [
        #{"grid_file": "mine_20x20.txt", "is_cnn": True, "model_name": "mine_20x20__reward_d_cnn_c4", "halfsplit": False},
        #{"grid_file": "a_30x30.txt", "is_cnn": True, "model_name": "a_30x30__reward_d_cnn_c5", "halfsplit": False},
        #{"grid_file": "b_30x30.txt", "is_cnn": True, "model_name": "b_30x30__reward_d_cnn_c5", "halfsplit": False},
        #{"grid_file": "c_30x30.txt", "is_cnn": True, "model_name": "c_30x30__reward_d_cnn_c5", "halfsplit": False},
        #{"grid_file": "a_50x50.txt", "is_cnn": False, "model_name": "a_50x50__reward_d", "halfsplit": False},
        #{"grid_file": "a_50x50.txt", "is_cnn": True, "model_name": "a_50x50__reward_d_cnn_unet", "halfsplit": False},
        #{"grid_file": "a_50x50.txt", "is_cnn": True, "model_name": "a_50x50__reward_d_cnn_c5", "halfsplit": False},
        #{"grid_file": "10p_100x100.txt", "is_cnn": True, "model_name": "10p_100x100__reward_d_cnn_c5", "halfsplit": False},
         #{"grid_file": "20p_100x100.txt", "is_cnn": True, "model_name": "20p_100x100__reward_d_cnn_c5", "halfsplit": False},
         #{"grid_file": "30p_100x100.txt", "is_cnn": True, "model_name": "30p_100x100__reward_d_cnn_c5", "halfsplit": False},
         #{"grid_file": "a_30x30.txt", "is_cnn": False, "model_name": "a_30x30__reward_e3_sigmoid", "halfsplit": False},
        #{"grid_file": "b_30x30.txt", "is_cnn": False, "model_name": "b_30x30__reward_e3_sigmoid", "halfsplit": False},
        #{"grid_file": "c_30x30.txt", "is_cnn": False, "model_name": "c_30x30__reward_e3_sigmoid", "halfsplit": False},
         #{"grid_file": "a_50x50.txt", "is_cnn": False, "model_name": "a_50x50__reward_e3_sigmoid", "halfsplit": False},
         {"grid_file": "mine_100x100.txt", "is_cnn": True, "model_name": "mine_100x100__reward_d_cnn_c6", "halfsplit": False},
    ]

    for config in models_to_train:
        try:
            print(f"\nTraining {config['model_name']} for {timesteps} timesteps on {config['grid_file']} (CNN={config['is_cnn']})")

            grid_path = os.path.join(FIXED_GRID_DIR, config["grid_file"])

            # compute battery overrides only if halfsplit flag is True
            battery_overrides = {}
            if config.get("halfsplit", False):
                battery_overrides = train.get_halfsplit_battery_overrides(grid_path)

            model = train.train_PPO_model(
                grid_file=config["grid_file"],
                timesteps=timesteps,
                reset_kwargs={"battery_overrides": battery_overrides} if battery_overrides else {},
                is_cnn=config["is_cnn"],
                folder_name=config["model_name"],
                battery_truncation=True
            )
        
            print(f"Finished training")
            
        except Exception as e:
            print(f"Failed to train {config['model_name']} on {config['grid_file']}: {e}")

def train_and_render_junk_model(grid_file: str = "mine_20x20.txt", is_cnn: bool = False, n_eval_episodes: int = 3):
    """
    Trains a quick junk PPO model and immediately renders it for evaluation.
    
    Args:
        grid_file (str): Filename of the grid in FIXED_GRID_DIR.
        is_cnn (bool): Whether to use CNN-based policy.
        n_eval_episodes (int): Number of episodes to evaluate with rendering.
    """
    print(f"\n[INFO] Training junk model on '{grid_file}' (CNN={is_cnn})...")
    
    # Train quick junk model and get experiment folder name
    experiment_base_folder = train.train_quick_junk_model(grid_file, is_cnn)
    base_path = os.path.join(SAVE_DIR, experiment_base_folder)
    latest_run_dir = get_latest_run_dir(base_path)
    
    print(f"[INFO] Loading junk model from '{latest_run_dir}'")

    # Load model from the saved experiment
    model = train.load_model(
        experiment_folder=latest_run_dir,
        grid_file=grid_file,
        is_cnn=is_cnn
    )

    # Prepare environment
    env = GridWorldEnv(grid_file=grid_file, is_cnn=is_cnn)
    if is_cnn:
        env = CustomGridCNNWrapper(env)

    print(f"[INFO] Rendering agent for {n_eval_episodes} episodes...\n")
    train.evaluate_model(
        env=env,
        model=model,
        n_eval_episodes=n_eval_episodes,
        render=True,
        verbose=True
    )
