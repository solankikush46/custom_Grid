# main.py

import os
import numpy as np
import json
import math

from src.constants import SAVE_DIR, FIXED_GRID_DIR
from src.SimulationController import *
from src.utils import *

def ensure_directories_exist():
    """Create necessary directories if they don't exist."""
    for d in (SAVE_DIR, FIXED_GRID_DIR):
        os.makedirs(d, exist_ok=True)

def render_test(experiment_folder = "mine_50x50_20miners"):
    """
    Initializes and runs the SimulationController. The controller itself
    will handle running simulations back-to-back.
    
    NOTE: This requires an experiment folder to exist that matches the
    specified name (e.g., 'saved_models/mine_50x50_20miners/').
    """
    print(f"[INFO] Initializing controller for experiment: '{experiment_folder}'")
    
    ctrl = SimulationController(
        experiment_folder=experiment_folder, 
        render=True, show_predicted=True,
        show_miners=True, mode="constant_rate",
    )
    
    # The run() method now contains the loop to run simulations continuously.
    # It will only return when the program is meant to shut down.
    ctrl.run()

def test_depletion_averages(n_episodes: int = 20,
                            experiment_folder: str = "mine_50x50_20miners"):
    """
    Runs `n_episodes` headless simulation episodes, collects each sensor's
    per-step battery-deltas, computes the average over all steps, and
    prints & saves the result.
    """
    print(f"[TEST] Collecting average battery deltas over {n_episodes} episodes for '{experiment_folder}'")
    
    # Headless, static mode, collect deltas
    ctrl = SimulationController(
        experiment_folder=experiment_folder,
        render=True,
        show_miners=True,
        show_predicted=False,
        mode="static",
        get_average_depletion=True,
        num_episodes=n_episodes
    )
    ctrl.run()  # blocks until num_episodes complete + shutdown()
    
    # Load the saved JSON (just to verify) and/or compute directly
    avg = ctrl.compute_average_depletion()
    
    print("\n[TEST] Average battery depletion per sensor (per timestep):")
    for pos, val in sorted(avg.items()):
        print(f"  Sensor {pos}: {val:.6f}")
    
    # JSON was saved to saved_experiments/<experiment_folder>/avg_sensor_depletion.json
    print(f"\n[TEST] Averages also written to: "
          f"saved_experiments/{experiment_folder}/avg_sensor_depletion.json")
    
def main():
    test_depletion_averages(2, "mine_1000x1000_80miners")
    render_test("mine_1000x1000_80miners")

if __name__ == "__main__":
    ensure_directories_exist()
    main()
