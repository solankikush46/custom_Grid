# main.py

import time
import os
from grid_env import GridWorldEnv
from train import *

def test_simple_grid():
    env = GridWorldEnv(
        grid_height=20,
        grid_width=20,
        n_obstacles=40,
        n_sensors=4
    )
    env.init_pygame()

    obs, _ = env.reset()
    done = False
    while not done:
        env.render_pygame()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.episode_summary()
    env.close()

def test_manual_control():
    env = GridWorldEnv(grid_height=20, grid_width=20, n_obstacles=20, n_sensors=3)
    env.manual_control_loop()

def test_simple_reward_20_20():
    env = GridWorldEnv(
        grid_height=20,
        grid_width=20,
        n_obstacles=0,
        n_sensors=0
    )
    env.init_pygame()

    obs, _ = env.reset()
    done = False
    while not done:
        env.render_pygame()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.episode_summary()
    env.close()
    
if __name__ == "__main__":
    #test_simple_grid()
    test_manual_control()
    #test_simple_reward_20_20()
    model = train_PPO_model(timesteps=100_000)
    evaluate_model(model)
