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
    '''
    # 20x20 with 15% obstacles and 3 sensors
    generate_grid(
    rows=20,
    cols=20,
    obstacle_percentage=0.15,
    n_sensors=3
    )

    # 20x20 with 30% obstacles and 3 sensors
    generate_grid(
    rows=20,
    cols=20,
    obstacle_percentage=0.30,
    n_sensors=3
    )
    '''
    
    #train_for_test_battery(50_000)
    #test_battery()
    #test_manual_control()
    test_PPO(timesteps=1_000_000, rows=20, cols=20)
