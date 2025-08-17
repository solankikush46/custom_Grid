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
    train_all(2_000)
    
if __name__ == "__main__":
    ensure_directories_exist()
    main()
