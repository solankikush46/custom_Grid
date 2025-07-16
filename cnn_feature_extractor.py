import gym
import numpy as np
import torch
import torch.nn as nn
from gym import ObservationWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

"""
Observation Wrappers : Outputs 4 channel grid for CNN
"""

class CustomGridCNNWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env =  env
        self.n_rows = env.n_rows
        self.n_cols = env.n_cols

        '''
        # 4 channels : agent_presence, blocked, sensor_battery, goal
        low = np.zeros((2, self.n_rows, self.n_cols), dtype=np.float32)
        high = np.array([4.0, 1.0]).reshape(2, 1, 1) * np.ones((2, self.n_rows, self.n_cols), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        '''

        low = np.zeros((5, self.n_rows, self.n_cols), dtype=np.float32)
        high = np.ones((5, self.n_rows, self.n_cols), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)


    def observation(self, obs):
        return obs


"""
CNN Feature Extractor for Stable-Baselines3
"""

import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GridCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, grid_file=None):
        super().__init__(observation_space, features_dim)
        self.grid_file = grid_file
        n_input_channels, height, width = observation_space.shape
        if grid_file and "20x20" in grid_file or "30x30" in grid_file:
            self.mode = "small"
            
            self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(40, 80, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(80, 160, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
            )
            '''
            self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
            )
            '''
            
        elif grid_file and "100x100" in grid_file:
            self.mode = "large"
            # Define layers for 100x100
            self.cnn = nn.Sequential(
                nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),     # 4 × 100 × 100 → 8 × 50 × 50
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),    # 8 × 50 × 50 → 16 × 25 × 25
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),   # 16 × 25 × 25 → 32 × 13 × 13
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # 32 × 13 × 13 → 64 × 7 × 7
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64 × 7 × 7 → 128 × 4 × 4
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 128 × 4 × 4 → 256 × 2 × 2
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 256 × 2 × 2 → 512 × 1 × 1
                nn.ReLU()
            )
            # Define linear layer etc. for large
        
        
        
        else:
            raise ValueError("Unknown grid size in filename!")

        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_input_channels, height, width)
            out = self.cnn(dummy_input)
            self.flattened_size = out.view(1, -1).shape[1]
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, features_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        x = self.cnn(obs)
        x = self.linear(x)
        return x
