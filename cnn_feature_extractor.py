import gym
import numpy as np
import torch
import torch.nn as nn
from gym import ObservationWrapper
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

class GridCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, grid_file=None, backbone="unet"):
        super().__init__(observation_space, features_dim)
        n_input_channels, height, width = observation_space.shape
        self.grid_file = grid_file
        self.backbone = backbone.lower()

        
        if grid_file and "20x20" in grid_file or "30x30" in grid_file:
            self.mode = "small"
            '''
            self.cnn = nn.Sequential(
            # Input: (5, 20, 20)
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
            # The uncommented, actual default:
            self.cnn = nn.Sequential(
                nn.Conv2d(5, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 12, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(24, 18, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(18, 36, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(36, 60, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(60, 100, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )

        elif grid_file and "100x100" in grid_file:
            self.mode = "large"
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

        elif grid_file and "50x50" in grid_file:
            self.mode = "custom-50x50"
            self.cnn = nn.Sequential(
                nn.Conv2d(5, 16, kernel_size=3, stride=2, padding=1),    
                nn.ReLU(),
                nn.Conv2d(16, 12, kernel_size=3, stride=1, padding=1),   
                nn.ReLU(),
                nn.Conv2d(12, 38, kernel_size=3, stride=2, padding=1),   
                nn.ReLU(),
                nn.Conv2d(38, 30, kernel_size=3, stride=1, padding=1),   
                nn.ReLU(),
                nn.Conv2d(30, 77, kernel_size=3, stride=2, padding=1),   
                nn.ReLU(),
                nn.Conv2d(77, 117, kernel_size=3, stride=2, padding=1),  
                nn.ReLU(),
                nn.Conv2d(117, 313, kernel_size=3, stride=2, padding=1), 
                nn.ReLU()
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


