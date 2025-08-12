# main.py

import os
import numpy as np
import json
import math

from src.constants import SAVE_DIR, FIXED_GRID_DIR
from src.SimulationController import *
from src.utils import *
from src.train import *

def ensure_directories_exist():
    """Create necessary directories if they don't exist."""
    for d in (SAVE_DIR, FIXED_GRID_DIR):
        os.makedirs(d, exist_ok=True)
    
def main():
    #test_depletion_averages(2, "mine_1000x1000_80miners")
    #render_test("mine_50x50_20miners")
    #train("mine_50x50_12miners", total_timesteps=1_000)
    train_junk("mine_50x50_12miners", render=True)
    #evaluate("mine_50x50_12miners", render=True)

if __name__ == "__main__":
    ensure_directories_exist()
    main()
