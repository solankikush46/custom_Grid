# main.py

import os
import numpy as np

from src.constants import SAVE_DIR, FIXED_GRID_DIR
from src.SimulationController import *
from src.utils import *

def ensure_directories_exist():
    """Create necessary directories if they don't exist."""
    for d in (SAVE_DIR, FIXED_GRID_DIR):
        os.makedirs(d, exist_ok=True)

def render_test():
    """
    Initializes and runs the SimulationController. The controller itself
    will handle running simulations back-to-back.
    
    NOTE: This requires an experiment folder to exist that matches the
    specified name (e.g., 'saved_models/mine_50x50_20miners/').
    """
    experiment_folder = "mine_50x50_20miners"
    
    print(f"[INFO] Initializing controller for experiment: '{experiment_folder}'")
    
    ctrl = SimulationController(
        experiment_folder=experiment_folder, 
        render=True, show_predicted=True, predicted_depletion_rate=0.712,
        show_miners=True, mode="constant_rate",
    )
    
    # The run() method now contains the loop to run simulations continuously.
    # It will only return when the program is meant to shut down.
    ctrl.run()

def main():
    #report_depletion_rate("mine_50x50_20miners", n_episodes=100)
    render_test()

if __name__ == "__main__":
    ensure_directories_exist()
    main()
