# main.py

from test import *
    
if __name__ == "__main__":
    #train_for_test_battery(50_000)
    #test_battery()
    #test_manual_control()
    test_PPO(timesteps=1_000_000, rows=20, cols=20)
    
