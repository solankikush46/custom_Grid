# test.py

import time
import os
from grid_env import GridWorldEnv
from constants import *
import train
import grid_gen

def test_manual_control():
    env = GridWorldEnv(grid_file="grid_20x20_30p.txt")
    env.manual_control_loop()    

def generate_grid(rows: int, cols: int, obstacle_percentage: float,
                  n_sensors: int=0, filename: str = None
                  ) -> str:
    '''
    Generates and saves a grid file, returning the filename
    '''
    if filename is None:
        filename = f"grid_{rows}x{cols}_{int(obstacle_percentage * 100)}p.txt"
    save_path = os.path.join(FIXED_GRID_DIR, filename)

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

def test_PPO(timesteps: int, rows: int, cols: int):
    episodes = 100
    render = True
    verbose = False
    
    # randomly generated
    obstacle_percentages = [] #[0.15, 0.30]

    for pct in obstacle_percentages:
        filename = f"grid_{rows}x{cols}_{int(pct * 100)}p.txt"
        model, name, env = train_model(filename, timesteps)
        '''
        train.load_model_and_evaluate(name, env,
                                      episodes,
                                      render=render,
                                      verbose=verbose)
        '''
        
    # fixed
    filename = f"mine_{rows}x{cols}.txt"
    model, name, env = train_model(filename, timesteps)
    '''
    train.load_model_and_evaluate(name, env,
                                      episodes,
                                      render=render,
                                      verbose=verbose)
    '''

def train_for_test_battery(timesteps: int):
    """
    Train PPO on the battery test grid using train_model helper
    """
    grid_filename = "battery_15x8.txt"

    env = GridWorldEnv(grid_file=grid_filename)

    obs, _ = env.reset(battery_overrides={
        (3, 0): 100.0,   # left sensor full
        (3, 11): 0.0     # right sensor empty
    })

    model_name = "battery_test"
    model = train.train_PPO_model(env, timesteps=timesteps, model_name=model_name)

    print("\nBattery test training complete.")
    return model, model_name, env

def test_fixed(filename: str, episodes: int = 10, render: bool = True, verbose: bool = False):
    """
    Load a model trained on the specified grid file and evaluate it.
    """
    model_name = os.path.splitext(filename)[0]

    env = GridWorldEnv(grid_file=filename)

    train.load_model_and_evaluate(
        model_filename=model_name,
        env=env,
        n_eval_episodes=episodes,
        sleep_time=0.1,
        render=render,
        verbose=verbose
    )

def test_battery():
    """
    Load the battery test model and evaluate using helper
    """
    grid_filename = "battery_15x8.txt"
    model_name = "battery_test"

    env = GridWorldEnv(grid_file=grid_filename)

    # override sensor batteries
    obs, _ = env.reset(battery_overrides={
        (3, 0): 100.0,   # left sensor full
        (3, 11): 0.0     # right sensor empty
    }, agent_override=[6, 7])

    train.load_model_and_evaluate(
        model_filename=model_name,
        env=env,
        n_eval_episodes=1,
        sleep_time=0.1,
        render=True,
        verbose=False
    )

