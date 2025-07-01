# main.py

from test import *
import os
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from grid_env import GridWorldEnv  # assuming your env is here

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
    test_PPO(500_000, 20, 20)

    # Create environment
    env = GridWorldEnv("mine_20x20.txt")  # customize as needed
    vec_env = DummyVecEnv([lambda: env])

    # Create PPO model
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./logs/ppo_gridworld/")

    # Train for 500,000 timesteps
    model.learn(total_timesteps=500_000)

    # Save model
    model.save("models/ppo_gridworld_500k")

    print("Training complete. Model saved to models/ppo_gridworld_500k.zip")

    
