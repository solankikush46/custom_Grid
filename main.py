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

def main():
    #generate_all_plots(rolling_window=2000)
    #test_PPO(5000, 100, 100)
    test_100x100_no_obstacles(timesteps=1_000_000, episodes=20)
        
if __name__ == "__main__":
    ensure_directories_exist()
    main()

    

