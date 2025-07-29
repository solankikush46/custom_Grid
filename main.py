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

def main():
    train_all_predictors()
    
if __name__ == '__main__':
    ensure_directories_exist()
    main()
