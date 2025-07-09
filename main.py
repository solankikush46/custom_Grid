# main.py

from test import *
from plot_metrics import generate_all_plots

def ensure_directories_exist():
    directories = [
        LOGS["ppo"],
        MODELS["ppo"],
        FIXED_GRID_DIR
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)

def main():
    #generate_all_plots(rolling_window=50_000)
    #test_PPO(10_000, 20, 20)
    #test_PPO(300_000, 100, 100)
    #test_100x100_no_obstacles(timesteps=500_000, episodes=20)
    #test_20x20_battery_override(1_000_000, 1)
    # Train and evaluate on mine_20x20 grid
    test_battery_half_split(
        grid_filename="mine_20x20.txt",
        timesteps=1_000_000,   # Adjust timesteps as needed
        episodes=5,          # Adjust evaluation episodes as needed
        render=False,
        verbose=True
    )

    # Train and evaluate on mine_100x100 grid
    test_battery_half_split(
        grid_filename="mine_100x100.txt",
        timesteps=1_000_000,  # More training steps for bigger grid
        episodes=5,
        render=False,
        verbose=True
    )
    #render_halfsplit_models()

if __name__ == "__main__":
    ensure_directories_exist()
    main()




    

