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
    '''
    TODO:
    - add all tensorboard metrics from prev version
    - read through rainbow paper to get ideas for improvements
    - get cnn results better than REU results that use D*Lite
    '''
    train_all(1_000_000)
    #evaluate_all(episodes=2, render=True, verbose=True, deterministic=False, exclude=["50x50"])
    
    #manual_control("mine_50x50__20miners__a1__reward_d")
    
if __name__ == "__main__":
    ensure_directories_exist()
    main()
