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

