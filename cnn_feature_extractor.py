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

        # 4 channels : agent_presence, blocked, sensor_battery, goal
        low = np.zeros((4, self.n_rows, self.n_cols), dtype=np.float32)
        high = np.ones((4, self.n_rows, self.n_cols), dtype=np.float32)

        # Adjust sensor battery channel to allow -1.0
        low[2, :, :] = -1.0  # Channel 2 (sensor battery)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        grid_tensor = np.zeros((4,self.n_rows, self.n_cols), dtype = np.float32)

        # Channel 0: agent_presence

        ar,ac = self.agent_pos
        grid_tensor[0, ar, ac] = 1.0

        # Channel 1: Blocked cells

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.env.static_grid[r,c] in ('#','S','B'):
                    grid_tensor[1,r,c] = 1.0

        # Channel 2: Sensor Battery Level

        grid_tensor[2, :, :] = -1.0 # default: not a sensor
        for (r,c), battery in self.env.sensor_batteries.items():
            grid_tensor[2,r,c] = battery/100.0

        # Channel 3 : Goal_cells
        for r,c in self.env.goal_positions:
            grid_tensor[3,r,c] = 1.0

        return grid_tensor

"""
CNN Feature Extractor for Stable-Baselines3
"""

class GridCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim =128):
        super(). __init__ (observation_space, features_dim=features_dim)
        C, H, W = observation_space.shape

        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, features_dim),
            nn.Tanh()
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.cnn(obs)

