# main.py

from test import *
from plot_metrics import generate_all_plots
from train import *

def ensure_directories_exist():
    directories = [
        SAVE_DIR,
        FIXED_GRID_DIR
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)

def main():
    #test_manual_control()
    #generate_all_plots(rolling_window=50_000)
    #test_PPO(10_000, 20, 20)
    #test_PPO(300_000, 100, 100)
    #test_100x100_no_obstacles(timesteps=500_000, episodes=20)
    #test_20x20_battery_override(1_000_000, 1)
    #evaluate_all_models()
    #train_all_halfsplit_models(1_000_000)
    #train_quick_junk_model("mine_20x20.txt")
    #test_render_junk_model("mine_20x20.txt")
    #train_and_render_junk_model()
    train_all_models(1_000_000)
    #evaluate_all_models(n_eval_episodes=20, render=True)
    
if __name__ == "__main__":
    ensure_directories_exist()
    main()

    

