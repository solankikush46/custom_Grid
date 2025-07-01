# main.py

from test import *
import os
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

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
    #test_PPO(timesteps=500_000, rows=20, cols=20)
    #test_manual_control("grid_20x20_30p.txt")
    test_PPO(1_000_000)
