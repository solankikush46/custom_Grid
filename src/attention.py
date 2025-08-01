# attention.py

from src.cnn_feature_extractor import *
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead = num_heads,
                dim_feedforward = embed_dim*2,
                dropout = dropout,
                activation = 'relu',
                batch_first = True
            ) for _ in range(num_layers)
        ])

        self.pool = nn.Linear(embed_dim,1)
        self.final_proj = nn.Linear(embed_dim, embed_dim)
        '''
        # Multihead attention over time. Input: (B, T, D)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Final projection back to embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)
        '''

    def forward(self, x_seq):  # x_seq: (B, T, D)
        '''
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)  # Self-attention across T time steps
        return self.proj(attn_out[:, -1])             # Only keep output from last time step
        '''
        for layer in self.layers:
            x_seq = layer(x_seq)

        attn_weights = torch.softmax(self.pool(x_seq), dim=1)  # (B, T, 1)
        pooled = (x_seq * attn_weights).sum(dim=1)             # (B, D)
        return self.final_proj(pooled)      

class AttentionCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, grid_file=None, temporal_len=4):
        T, C, H, W = observation_space.shape
        super().__init__(observation_space, features_dim)
        self.cnn = build_default_cnn(C, grid_file)
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            cnn_out = self.cnn(dummy)
            cnn_out = cnn_out.view(1, -1)
            self.cnn_feature_dim = cnn_out.shape[1]
        self.temporal_att = TemporalAttention(
            embed_dim=self.cnn_feature_dim,
            num_heads=2,         # Try 2 or 4
            num_layers=2,        # Try 2 or 3
            dropout=0.1
            )
        self.out = nn.Sequential(
            nn.Linear(self.cnn_feature_dim, features_dim),
            nn.Tanh()
        )
        
    def forward(self, obs):  # obs: (B, T, C, H, W)
        B, T, C, H, W = obs.shape
        obs = obs.view(B * T, C, H, W)
        x = self.cnn(obs)
        x = x.view(B, T, -1)
        x = self.temporal_att(x)
        return self.out(x)

class SpatialChannelAttention(nn.Module):
    """
    Improved CBAM with residual, stronger channel & spatial blocks,
    and optional use of coordinate channels (for grid tasks).
    """
    def __init__(self, in_channels, height, width, reduction=16, use_coords=True):
        super().__init__()
        self.use_coords = use_coords

        # Channel attention (SE block)
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial attention (CBAM spatial)
        spatial_in_channels = 2 + (2 if use_coords else 0)  # avg + max + (x,y)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(spatial_in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        # Precompute coordinate maps if needed
        if use_coords:
            yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij")
            self.register_buffer("coord_x", xx.unsqueeze(0).unsqueeze(0))  # (1,1,H,W)
            self.register_buffer("coord_y", yy.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # Channel attention
        ch_att = self.channel_fc(x)
        x_ca = x * ch_att

        # Spatial attention (use both CA output and original features for strong residual)
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_input = [avg_out, max_out]
        if self.use_coords:
            spatial_input += [self.coord_x.expand(x.size(0), -1, -1, -1), self.coord_y.expand(x.size(0), -1, -1, -1)]
        spatial_in = torch.cat(spatial_input, dim=1)
        sp_att = self.spatial_conv(spatial_in)
        x_out = x_ca * sp_att

        # Residual: add input back in (stabilizes gradients, helps small-data)
        return x_out + x