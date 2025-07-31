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
    CBAM-inspired channel and spatial attention for feature maps.
    - Channel attention focuses on "what" is important.
    - Spatial attention focuses on "where" is important.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # ---- Channel attention (Squeeze-and-Excitation style) ----
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # ---- Spatial attention ----
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Ensure x is 4D: [B, C, H, W]
        if x.dim() == 2:
            # Example: [batch, channels] → [batch, channels, 1, 1]
            x = x.unsqueeze(-1).unsqueeze(-1)
        elif x.dim() == 3:
            # Example: [batch, channels, H] → [batch, channels, H, 1]
            x = x.unsqueeze(-1)

        # --- Channel attention ---
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.channel_mlp(avg_out)
        max_out = self.channel_mlp(max_out)
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        # --- Spatial attention ---
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att

        return x
