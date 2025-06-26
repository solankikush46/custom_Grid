# main.py

import time
import os
from grid_env import GridWorldEnv
from train import *
import grid_gen

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
    env = GridWorldEnv(grid_file="grid_20x20_15p.txt")
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

def test_simple_PPO():
    env = GridWorldEnv(
        grid_file="grid_20x20.txt"
    )

    model = train_PPO_model(env, timesteps=100_000)
    load_model_and_evaluate("ppo", env, 20, render=True, verbose=True)

def test_PPO_20x20():
    filename_15 = "grid_20x20_15p.txt"
    save_path_15 = os.path.join(FIXED_GRID_DIR, filename_15)

    grid_gen.gen_and_save_grid(
    rows=20,
    cols=20,
    obstacle_percentage=0.05,
    n_sensors=0,
    place_agent=True,
    save_path=save_path_15
    )

    env_15 = GridWorldEnv(grid_file=filename_15)
    # model_15 = train_PPO_model(env_15, timesteps=100_000)
    load_model_and_evaluate("ppo", env_15, 20, render=True, verbose=True)

    # === Generate and train on 30% obstacle grid ===
    filename_30 = "grid_20x20_30p.txt"
    save_path_30 = os.path.join(FIXED_GRID_DIR, filename_30)

    '''
    grid_gen.gen_and_save_grid(
    rows=20,
    cols=20,
    obstacle_percentage=0.30,
    n_sensors=5,
    place_agent=True,
    save_path=save_path_30
    )
    '''

    env_30 = GridWorldEnv(grid_file=filename_30)
    # model_30 = train_PPO_model(env_30, timesteps=1_000_000)
    load_model_and_evaluate("ppo", env_30, 20, render=True, verbose=True)
    
if __name__ == "__main__":
    #test_simple_grid()
    #test_manual_control()
    #test_simple_reward_20_20()
    #test_simple_PPO()
    #model = train_PPO_model(timesteps=100_000)
    #evaluate_model(model)
    test_PPO_20x20()
    
