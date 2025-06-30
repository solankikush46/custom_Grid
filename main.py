# main.py

from test import *
import os

def ensure_directories_exist():
    directories = [
        LOGS["ppo"],
        LOGS["dqn"],
        MODELS["ppo"],
        MODELS["dqn"],
        FIXED_GRID_DIR,
        RANDOM_GRID_DIR,
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)
    
if __name__ == "__main__":
    ensure_directories_exist()
    #train_for_test_battery(50_000)
    #test_battery()
    #test_manual_control()
    #test_PPO(timesteps=400_000, rows=20, cols=20)
    test_fixed("mine_20x20.txt", episodes=1000, render=True, verbose=False)
    
