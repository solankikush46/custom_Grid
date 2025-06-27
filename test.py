# test.py

import time
import os
from grid_env import GridWorldEnv
from constants import *
import train
import grid_gen

def test_manual_control():
    env = GridWorldEnv(grid_file="grid_20x20_15p.txt")
    env.manual_control_loop()    

def generate_grid(rows: int, cols: int, obstacle_percentage: float,
                  n_sensors: int=0, filename: str = None
                  ) -> str:
    '''
    Generates and saves a grid file, returning the filename
    '''
    if filename is None:
        filename = f"grid_{rows}x{cols}_{int(obstacle_percentage * 100)}p.txt"
    save_path = os.path.join(RANDOM_GRID_DIR, filename)

    grid_gen.gen_and_save_grid(
        rows=rows,
        cols=cols,
        obstacle_percentage=obstacle_percentage,
        n_sensors=n_sensors,
        place_agent=False,
        save_path=save_path
    )

    return filename

def train_model(filename: str, timesteps: int):
    """
    Loads the environment and trains the PPO model
    """
    env = GridWorldEnv(grid_file=filename)
    model_name = os.path.splitext(filename)[0]
    model = train.train_PPO_model(env, timesteps=timesteps,
                                  model_name=model_name)
    return model, model_name, env

def test_PPO_20x20(timesteps: int):
    episodes = 100
    render = True
    verbose = False
    obstacle_percentages = [0.15, 0.30]

    for pct in obstacle_percentages:
        filename = generate_grid(rows=20, cols=20,
                                 obstacle_percentage=pct)
        model, name, env = train_model(filename, timesteps)
        train.load_model_and_evaluate(name, env,
                                      episodes,
                                      render=render,
                                      verbose=verbose)

