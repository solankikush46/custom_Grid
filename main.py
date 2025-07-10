# main.py

from test import *
from plot_metrics import generate_all_plots
from train import *

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
    #generate_all_plots(rolling_window=50_000)
    #test_PPO(10_000, 20, 20)
    #test_PPO(300_000, 100, 100)
    #test_100x100_no_obstacles(timesteps=500_000, episodes=20)
    #test_20x20_battery_override(1_000_000, 1)
    create_and_train_cnn_halfsplit_model("mine_20x20.txt", 1_000_000)
    create_and_train_cnn_halfsplit_model("mine_100x100.txt", 1_000_000)

if __name__ == "__main__":
    ensure_directories_exist()
    main()




    

