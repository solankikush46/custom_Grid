# cnn_feature_extractor.py

import gym
import numpy as np
import torch
import torch.nn as nn
from gym import ObservationWrapper, spaces
from utils import get_c8_neighbors_status, make_agent_feature_matrix
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F


"""
Observation Wrappers : Outputs 4 channel grid for CNN
"""

class CustomGridCNNWrapper(ObservationWrapper):
    """
    Ensures observations are numpy arrays with dtype float32
    and are channel-first (C, H, W) for CNN feature extractors.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space  # Already set properly in your env

    def observation(self, observation):
        # If it's not already a numpy array, make it one
        obs = np.array(observation, dtype=np.float32)
        # If needed, add further checks for channel order here
        return obs

class UNetPathfinder(nn.Module):
    def __init__(self, input_channels=5, base_filters=32):
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(input_channels, base_filters)
        self.enc2 = self.conv_block(base_filters, base_filters*2)
        self.enc3 = self.conv_block(base_filters*2, base_filters*4)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*8, 3, padding=1),
            nn.BatchNorm2d(base_filters*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters*8, base_filters*8, 3, padding=1),
            nn.BatchNorm2d(base_filters*8),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.dec3 = self.upconv_block(base_filters*8, base_filters*4)
        self.dec2 = self.upconv_block(base_filters*4*2, base_filters*2)  # *2 for skip connection
        self.dec1 = self.upconv_block(base_filters*2*2, base_filters)

        # Output
        self.final_conv = nn.Conv2d(base_filters*2, 1, kernel_size=1)  # *2 for last skip connection

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),  # upsample
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e3, 2))

        # Decoder
        d3 = self.dec3(F.interpolate(b, scale_factor=2, mode='nearest'))
        # Align shapes for skip connection
        if d3.shape[2:] != e3.shape[2:]:
            e3_resized = F.interpolate(e3, size=d3.shape[2:], mode='nearest')
        else:
            e3_resized = e3
        d3 = torch.cat([d3, e3_resized], dim=1)  # Skip connection

        d2 = self.dec2(F.interpolate(d3, scale_factor=2, mode='nearest'))
        if d2.shape[2:] != e2.shape[2:]:
            e2_resized = F.interpolate(e2, size=d2.shape[2:], mode='nearest')
        else:
            e2_resized = e2
        d2 = torch.cat([d2, e2_resized], dim=1)

        d1 = self.dec1(F.interpolate(d2, scale_factor=2, mode='nearest'))
        if d1.shape[2:] != e1.shape[2:]:
            e1_resized = F.interpolate(e1, size=d1.shape[2:], mode='nearest')
        else:
            e1_resized = e1
        d1 = torch.cat([d1, e1_resized], dim=1)

        out = self.final_conv(d1)
        return torch.sigmoid(out)
class UNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, input_channels=5, base_filters=32):
        super().__init__(observation_space, features_dim)
        self.unet = UNetPathfinder(input_channels=input_channels, base_filters=base_filters)
        # Figure out flattened size by passing a dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *observation_space.shape[1:])
            unet_out = self.unet(dummy)
            n_flat = unet_out.view(1, -1).shape[1]
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, features_dim),
            nn.ReLU()
        )

class GridCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, grid_file=None, backbone="cnn"):
        super().__init__(observation_space, features_dim)
        n_input_channels, height, width = observation_space.shape
        self.grid_file = grid_file
        self.backbone = backbone.lower()

        
        if grid_file and "20x20" in grid_file or "30x30" in grid_file:
            self.mode = "small"
            '''
            #c4 architecture
            self.cnn = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, stride=2, padding=1),   # (16, 10, 10) ~80%
            nn.ReLU(),
            nn.Conv2d(16, 12, kernel_size=3, stride=1, padding=1),  # (12, 10, 10) ~60%
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),  # (24, 5, 5)  ~50%
            nn.ReLU(),
            nn.Conv2d(24, 18, kernel_size=3, stride=1, padding=1),  # (18, 5, 5)  ~40%
            nn.ReLU(),
            nn.Conv2d(18, 36, kernel_size=3, stride=2, padding=1),  # (36, 3, 3)  ~30%
            nn.ReLU(),
            nn.Conv2d(36, 60, kernel_size=3, stride=1, padding=1),  # (60, 3, 3)  ~15%
            nn.ReLU(),
            nn.Conv2d(60, 100, kernel_size=3, stride=2, padding=1), # (100, 2, 2) ~10%
            nn.ReLU()
            )
            '''
            '''
            #c3 architecture
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

            '''
            #c2 architecture
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
            # c5 architecture
            self.cnn = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1),   # (32, 15, 15)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (256, 2, 2)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.AdaptiveAvgPool2d((1, 1)),  # (256, 1, 1)
            nn.Flatten()
            )

        elif grid_file and "100x100" in grid_file:
            self.mode = "large"
            '''
            # c3 architecture
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
            '''
            # c5 architecture

            self.cnn = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1),    # (32, 50, 50)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # (64, 25, 25)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 13, 13)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (256, 7, 7)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # (256, 2, 2)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),  # (256, 1, 1)
            nn.Flatten()
            )
            '''
            #c6 architecture
            self.cnn = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=7, stride=2, padding=3),    # (32, 50, 50)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),   # (64, 25, 25)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, 13, 13)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (256, 7, 7)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # (256, 2, 2)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),  # (256, 1, 1)
            nn.Flatten()
        )
        elif grid_file and "50x50" in grid_file:
            self.mode = "custom-50x50"
            #c5 architecture
            self.cnn = nn.Sequential(
                nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1),   # (32, 25, 25)
                nn.BatchNorm2d(32),
                nn.ReLU(),

                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 13, 13)
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout2d(0.1),

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (128, 7, 7)
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (256, 4, 4)
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Dropout2d(0.1),

                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # (256, 2, 2)
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.AdaptiveAvgPool2d((1, 1)),  # (256, 1, 1)
                nn.Flatten()
            )
            '''
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
        else:
            raise ValueError("Unknown grid size in filename!")

        # Calculate flattened size for CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_input_channels, height, width)
            cnn_out = self.cnn(dummy_input)
            self.cnn_flattened_size = cnn_out.view(1, -1).shape[1]
        self.cnn_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.cnn_flattened_size, features_dim),
            nn.Tanh()
        )

        # ----------------- U-Net backbone -----------------
        self.unet = UNetPathfinder(input_channels=n_input_channels, base_filters=32)
        with torch.no_grad():
            unet_out = self.unet(dummy_input)
            self.unet_flattened_size = unet_out.view(1, -1).shape[1]
        self.unet_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.unet_flattened_size, features_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        if self.backbone == "unet":
            unet_out = self.unet(obs)            # [batch, 1, H, W]
            features = self.unet_linear(unet_out)
            return features
        elif self.backbone == "cnn":
            cnn_out = self.cnn(obs)              # [batch, C, H', W']
            features = self.cnn_linear(cnn_out)
            return features
        else:
            raise ValueError(f"Unknown backbone type: {self.backbone}")


class AgentFeatureMatrixWrapper(ObservationWrapper):
    """
    Replaces observation with a 5x5 feature matrix per timestep.
    """
    def __init__(self, env, max_sensors=9):
        super().__init__(env)
        self.max_sensors = max_sensors
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5, 5), dtype=np.float32
        )

    def observation(self, obs):
        agent_pos = self.env.agent_pos
        grid = self.env.grid
        last_action = getattr(self.env, 'last_action', 0)
        goals = self.env.goal_positions
        goal_dist = min([abs(agent_pos[0] - gx) + abs(agent_pos[1] - gy) for (gx, gy) in goals])
        neighbors = get_c8_neighbors_status(grid, agent_pos, obstacle_val=self.env.OBSTACLE_VALS)
        sensor_batteries = [
            self.env.sensor_batteries[pos]
            for pos in sorted(self.env.sensor_batteries.keys())
        ][:self.max_sensors]
        sensor_batteries += [0.0] * (self.max_sensors - len(sensor_batteries))
        return make_agent_feature_matrix(agent_pos, neighbors, last_action, goal_dist, sensor_batteries, self.max_sensors)
    
class FeatureMatrixCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        n_channels, height, width = 1, *observation_space.shape  # (1, 5, 5)
        self.cnn = nn.Sequential(
            # First conv: out (8, 3, 3)
            nn.Conv2d(n_channels, 8, kernel_size=3, stride=1, padding=0),   # (5-3)/1+1=3
            nn.ReLU(),
            # Second conv: out (16, 1, 1)
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),           # (3-3)/1+1=1
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, height, width)
            n_flat = self.cnn(dummy).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        if obs.dim() == 3:
            obs = obs.unsqueeze(1)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(0).unsqueeze(0)
        x = self.cnn(obs)
        return self.linear(x)