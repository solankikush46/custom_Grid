# main.py

from src.test import *
from src.plot_metrics import generate_all_plots
from src.train import *
from src.MineSimulator import *

def ensure_directories_exist():
    directories = [
        SAVE_DIR,
        FIXED_GRID_DIR
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)

import random
from src.MineSimulator import MineSimulator
from src.constants import FIXED_GRID_DIR

import random
from src.MineSimulator import MineSimulator
from src.constants import FIXED_GRID_DIR

def run_simulation_test():
    """
    Tests the simulator, which now manages its own rendering internally.
    """
    print("--- Starting Simulation Test with Internal Renderer ---")
    try:
        grid_filename = "mine_20x20.txt"
        
        # 1. Create the simulator and tell it you want to see the graphics
        simulator = MineSimulator(grid_file=grid_filename, n_miners=10, render_mode="human")

    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        return

    # Reset returns the initial state
    initial_state = simulator.reset()
    print(f"Simulator Initialized. Agent starts at: {initial_state['agent_pos']}")

    running = True
    while running:
        # Game Logic
        random_action = random.randint(0, 7)
        simulator.step(agent_action=random_action)

        # Rendering is now a simple, self-contained call
        # It returns False if the user quits
        running = simulator.render()

    # The simulator's close method will handle shutting down pygame
    simulator.close()
    print("--- Simulation Test Finished ---")

def main():
    #test_manual_control("mine_20x20.txt")
    #generate_all_plots(rolling_window=50_000)
    #train_all_models(1_000_000)
    #evaluate_all_models(n_eval_episodes=10000, render=True, verbose=True, dos=["higher_alpha"])
    run_simulation_test()
    
if __name__ == "__main__":
    ensure_directories_exist()
    main()

