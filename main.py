# main.py

from test import *
from plot_metrics import generate_all_plots

def ensure_directories_exist():
    directories = [
        LOGS["ppo"],
        MODELS["ppo"],
        FIXED_GRID_DIR,
        RANDOM_GRID_DIR,
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)

def main():
    generate_all_plots()
    #test_PPO(1_000_000)
        
if __name__ == "__main__":
    ensure_directories_exist()
    main()

    

