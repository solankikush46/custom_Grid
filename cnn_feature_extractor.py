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
        return obs


"""
CNN Feature Extractor for Stable-Baselines3
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GridCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input_channels, height, width = observation_space.shape

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Automatically determine flattened size
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

