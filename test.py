# test.py

import time
import os
from grid_env import GridWorldEnv
from constants import *
import train
import grid_gen
from cnn_feature_extractor import CustomGridCNNWrapper, GridCNNExtractor

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

def train_model(filename: str, timesteps: int):
    model_name = os.path.splitext(filename)[0]
    model = train.train_PPO_model(grid_file=filename,
                                  timesteps=timesteps,
                                  model_name=model_name)
    eval_env = GridWorldEnv(grid_file=filename)
    return model, model_name, eval_env

def test_PPO(timesteps: int, rows: int, cols: int):
    scenarios = [
        "grid_{rows}x{cols}_15p.txt",
        "grid_{rows}x{cols}_30p.txt",
        "mine_{rows}x{cols}.txt",
    ]

    for template in scenarios:
        filename = template.format(rows=rows, cols=cols)
        label = filename.rsplit(".", 1)[0]  # remove .txt (1 means max of 1 split)
        train.train_PPO_model(filename, timesteps, label)

def train_for_test_battery(timesteps: int):
    """
    Train PPO on the battery test grid using train_model helper
    """
    grid_filename = "battery_15x8.txt"

    env = GridWorldEnv(grid_file=grid_filename)

    obs, _ = env.reset(battery_overrides={
        (3, 0): 100.0,   # left sensor full
        (3, 11): 0.0     # right sensor empty
    })

    model_name = "battery_test"
    model = train.train_PPO_model(env, timesteps=timesteps, model_name=model_name)

    print("\nBattery test training complete.")
    return model, model_name, env

def test_fixed(filename: str, episodes: int = 10, render: bool = True, verbose: bool = False):
    """
    Load a trained model for the specified grid file and evaluate it.
    If no trained model exists, run evaluation with random actions.
    """
    model_name = os.path.splitext(filename)[0]
    env = GridWorldEnv(grid_file=filename)

    model_path = os.path.join(train.MODELS["ppo"], model_name + ".zip")

    if os.path.exists(model_path):
        if verbose:
            print(f"Loading model '{model_name}' for evaluation.")
        model = PPO.load(model_path, env=DummyVecEnv([lambda: env]))
        # Evaluate the model
        train.evaluate_model(env, model, n_eval_episodes=episodes, sleep_time=0.1, render=render, verbose=verbose)
    else:
        if verbose:
            print(f"No trained model found for '{model_name}'. Running random policy evaluation.")
        # Evaluate random policy
        total_rewards = []
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated
                if render:
                    env.render_pygame()
                    time.sleep(0.1)
            total_rewards.append(ep_reward)

        mean_reward = sum(total_rewards) / episodes
        std_reward = (sum((r - mean_reward) ** 2 for r in total_rewards) / episodes) ** 0.5
        print(f"\nRandom policy evaluation over {episodes} episodes:")
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

def test_battery():
    """
    Load the battery test model and evaluate using helper
    """
    grid_filename = "battery_15x8.txt"
    model_name = "battery_test"

    env = GridWorldEnv(grid_file=grid_filename)

    # override sensor batteries
    obs, _ = env.reset(battery_overrides={
        (3, 0): 100.0,   # left sensor full
        (3, 11): 0.0     # right sensor empty
    }, agent_override=[6, 7])

    train.load_model_and_evaluate(
        model_filename=model_name,
        env=env,
        n_eval_episodes=1,
        sleep_time=0.1,
        render=True,
        verbose=False
    )

def test_100x100_no_obstacles(timesteps: int = 5000, episodes: int = 5, render: bool = True, verbose: bool = True):
    """
    Generates a 100x100 grid with 0% obstacles, trains PPO, and evaluates it.
    """
    # generate grid filename
    filename = generate_grid(rows=100, cols=100, obstacle_percentage=0.0)

    # train the model
    if verbose:
        print(f"\nTraining PPO model on '{filename}' for {timesteps} timesteps...")
    model, model_name, eval_env = train_model(filename=filename, timesteps=timesteps)

    # evaluate the trained model
    if verbose:
        print(f"\nEvaluating trained model '{model_name}' for {episodes} episodes...")
    train.evaluate_model(
        env=eval_env,
        model=model,
        n_eval_episodes=episodes,
        sleep_time=0.1,
        render=render,
        verbose=verbose
    )

def test_20x20_battery_override(timesteps: int, episodes: int = 3, render: bool = True, verbose: bool = True):
    """
    Loads a fixed 20x20 map, overrides bottom 2 sensor batteries to 0.0, others to 100.0,
    then trains and evaluates PPO on this scenario.
    """
    filename = "mine_20x20.txt"
    filepath = os.path.join(FIXED_GRID_DIR, filename)

    # --- Step 1: Parse sensor positions from file
    sensor_positions = []
    with open(filepath, "r") as f:
        for r, line in enumerate(f):
            for c, char in enumerate(line.strip()):
                if char == "S":
                    sensor_positions.append((r, c))

    if len(sensor_positions) < 2:
        raise ValueError("Not enough sensors in the grid to run this test.")

    # Sort by row descending (bottom sensors last)
    sensor_positions_sorted = sorted(sensor_positions, key=lambda x: x[0], reverse=True)

    # --- Step 2: Build battery override dictionary
    battery_overrides = {}
    bottom_two = sensor_positions_sorted[:2]
    rest = sensor_positions_sorted[2:]

    for pos in bottom_two:
        battery_overrides[pos] = 0.0
    for pos in rest:
        battery_overrides[pos] = 100.0

    if verbose:
        print("Battery overrides:")
        for k, v in battery_overrides.items():
            print(f"  Sensor at {k}: battery = {v}")

    # --- Step 3: Create environment and train
    env = GridWorldEnv(grid_file=filename)
    obs, _ = env.reset(battery_overrides=battery_overrides)

    model_name = "battery_override_20x20"
    model = train.train_PPO_model(grid_file=filename, timesteps=timesteps, model_name=model_name)

    # --- Step 4: Evaluate using overides
    reset_kwargs = {"battery_overrides": battery_overrides}

    train.load_model_and_evaluate(
        model_filename=model_name,
        env=env,
        n_eval_episodes=episodes,
        sleep_time=0.1,
        render=render,
        verbose=verbose,
        reset_kwargs=reset_kwargs
    )

def load_and_evaluate_battery_override_model(
    episodes: int = 3,
    render: bool = True,
    verbose: bool = True,
    sleep_time: float = 0.1
):
    """
    Loads the battery override PPO model created by test_20x20_battery_override,
    recreates the environment with the same battery overrides, and evaluates the model.
    """

    filename = "mine_20x20.txt"
    filepath = os.path.join(FIXED_GRID_DIR, filename)

    # --- Parse sensor positions from file (same as in test_20x20_battery_override)
    sensor_positions = []
    with open(filepath, "r") as f:
        for r, line in enumerate(f):
            for c, char in enumerate(line.strip()):
                if char == "S":
                    sensor_positions.append((r, c))

    if len(sensor_positions) < 2:
        raise ValueError("Not enough sensors in the grid to run this test.")

    # Sort by row descending (bottom sensors last)
    sensor_positions_sorted = sorted(sensor_positions, key=lambda x: x[0], reverse=True)

    # Build battery overrides dict
    battery_overrides = {}
    bottom_two = sensor_positions_sorted[:2]
    rest = sensor_positions_sorted[2:]

    for pos in bottom_two:
        battery_overrides[pos] = 0.0
    for pos in rest:
        battery_overrides[pos] = 100.0

    # Create environment
    env = GridWorldEnv(grid_file=filename)

    # Model name used during training
    model_name = "battery_override_20x20"

    # Evaluate using train.py helper
    train.load_model_and_evaluate(
        model_filename=model_name,
        env=env,
        n_eval_episodes=episodes,
        sleep_time=sleep_time,
        render=render,
        verbose=verbose,
        reset_kwargs={"battery_overrides": battery_overrides}
    )

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

def test_battery_half_split(grid_filename: str, timesteps: int,
                            episodes: int = 3, render: bool = True, verbose: bool = True):
    """
    Full pipeline: generate battery overrides, train model, and evaluate on the battery half-split scenario.
    Battery overrides are generated once and passed through.
    """
    grid_path = os.path.join(FIXED_GRID_DIR, grid_filename)
    battery_overrides = train.get_halfsplit_battery_overrides(grid_path)

    if verbose:
        print("Battery overrides for training and evaluation:")
        for k, v in battery_overrides.items():
            print(f"  Sensor at {k}: battery = {v}")

    model_name, _ = train.train_halfsplit_model(grid_filename, timesteps, battery_overrides)
    train.evaluate_halfsplit_model(model_name, grid_filename, battery_overrides,
                            episodes=episodes, render=render, verbose=verbose)

def test_halfsplit_model(grid_file, episodes):
    model_name = f"battery_halfsplit_{grid_file.replace('.txt','')}" + "1"
    grid_path = os.path.join(FIXED_GRID_DIR, grid_file)
    battery_overrides = train.get_halfsplit_battery_overrides(grid_path)

    train.evaluate_halfsplit_model(
        model_name=model_name,
        grid_filename=grid_file,
        battery_overrides=battery_overrides,
        episodes=20,
        render=True,
        verbose=True
    )

def render_halfsplit_models():
    # Reconstruct battery overrides
    grid_20 = "mine_20x20.txt"
    grid_100 = "mine_100x100.txt"

    path_20 = os.path.join(FIXED_GRID_DIR, grid_20)
    path_100 = os.path.join(FIXED_GRID_DIR, grid_100)

    overrides_20 = train.get_halfsplit_battery_overrides(path_20)
    overrides_100 = train.get_halfsplit_battery_overrides(path_100)

    # Evaluate the previously saved models
    train.evaluate_halfsplit_model(
        model_name="battery_halfsplit_mine_20x201",
        grid_filename=grid_20,
        battery_overrides=overrides_20,
        episodes=100,
        render=True,
        verbose=True
    )

    '''
    train.evaluate_halfsplit_model(
        model_name="battery_halfsplit_mine_100x100",
        grid_filename=grid_100,
        battery_overrides=overrides_100,
        episodes=5,
        render=True,
        verbose=True
    )
    '''

def evaluate_all_models():
    """
    Automatically evaluates and renders all PPO models in SavedModels/PPO_custom_grid.
    Infers CNN/MLP, grid file, and battery override from filename.
    """
    model_dir = MODELS["ppo"]
    model_files = [
        f for f in os.listdir(model_dir)
        if f.endswith(".zip") and os.path.isfile(os.path.join(model_dir, f))
    ]

    for model_file in model_files:
        model_name = model_file.replace(".zip", "")

        # Determine CNN or MLP
        is_cnn = "cnn" in model_name

        # Infer grid filename
        if "100x100" in model_name:
            grid_filename = "mine_100x100.txt"
        elif "20x20" in model_name:
            grid_filename = "mine_20x20.txt"
        else:
            print(f"Skipping {model_name}: cannot infer grid file.")
            continue

        # Infer battery overrides
        reset_kwargs = {}
        if "halfsplit" in model_name:
            grid_path = os.path.join(FIXED_GRID_DIR, grid_filename)
            battery_overrides = train.get_halfsplit_battery_overrides(grid_path)
            reset_kwargs["battery_overrides"] = battery_overrides

        print(f"\n=== Evaluating {model_name} ===")
        print(f"Grid: {grid_filename} | CNN: {is_cnn} | Halfsplit: {'halfsplit' in model_name}")

        train.load_model_and_evaluate(
            model_filename=model_name,
            grid_file=grid_filename,
            is_cnn=is_cnn,
            reset_kwargs=reset_kwargs,
            n_eval_episodes=3,
            sleep_time=0.1,
            render=True,
            verbose=True
        )

def test_render_junk_model(is_cnn=False):
    # Example usage
    grid_file = "mine_20x20.txt"  # or whatever grid you want to test
    
    # Train junk model
    model_name = train.train_quick_junk_model(grid_file, is_cnn)
    
    # Load env for evaluation with same grid
    env = GridWorldEnv(grid_file=grid_file, is_cnn=is_cnn)
    if is_cnn:
        env = CustomGridCNNWrapper(env)
    
    model_path = os.path.join(MODELS["ppo"], model_name)
    model = train.load_model(model_path, grid_file=grid_file, is_cnn=is_cnn)
    
    # Evaluate with rendering enabled to watch agent behavior
    train.evaluate_model(env, model, n_eval_episodes=3, render=True, verbose=True)

