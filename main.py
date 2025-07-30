# main.py

import os
from src.constants import *
from src.SimulationController import *

def ensure_directories_exist():
    """Create necessary directories if they don't exist."""
    directories = [
        SAVE_DIR,
        FIXED_GRID_DIR
    ]
    for d in directories:
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
    print("The controller will run simulations continuously. Close the render window to exit.")
    
    ctrl = SimulationController(
        experiment_folder=experiment_folder, 
        render=True
    )
    
    # The run() method now contains the loop to run simulations continuously.
    # It will only return when the program is meant to shut down.
    ctrl.run()
    
def main():
    """Main function to start the simulation runner."""
    render_test()
    
if __name__ == '__main__':
    ensure_directories_exist()
    main()
