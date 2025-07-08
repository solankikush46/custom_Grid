# main.py

from test import *
from plot_metrics import generate_all_plots

def ensure_directories_exist():
    directories = [
        LOGS["ppo"],
        MODELS["ppo"],
        FIXED_GRID_DIR,
        RANDOM_GRID_DIR,
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)

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

def main():
    #generate_all_plots(rolling_window=50_000)
    #test_PPO(10_000, 20, 20)
    #test_PPO(300_000, 100, 100)
    #test_100x100_no_obstacles(timesteps=500_000, episodes=20)
    test_20x20_battery_override(1_000_000, 1)
    #test_manual_control()
    #load_and_evaluate_battery_override_model(episodes=20, render=True, verbose=True)
    #simulate_battery_depletion()
    #test_observation_space()
    
if __name__ == "__main__":
    ensure_directories_exist()
    main()



    

