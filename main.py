# main.py

import random
from src.MineSimulator import MineSimulator
from src.constants import *
from src.SimulationController import *
from src.train import *

def ensure_directories_exist():
    directories = [
        SAVE_DIR,
        FIXED_GRID_DIR
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)

def render_test():
    """
    Runs a quick rendering test using the SimulationController.
    Modify the grid file name and miner count as needed.
    """
    grid_file = "mine_50x50.txt"   # Make sure this file exists in saved_grids/fixed
    n_miners = 20                  # Number of autonomous miners to simulate
    print("[TEST] Starting render test with SimulationController...")
    
    ctrl = SimulationController(grid_file=grid_file, n_miners=n_miners, render=True)
    ctrl.run()

def main():
    #train_all_predictors()
    while True:
        render_test()
    
if __name__ == '__main__':
    ensure_directories_exist()
    main()
