# attention.py
from src.cnn_feature_extractor import *
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()

        # Multihead attention over time. Input: (B, T, D)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Final projection back to embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_seq):  # x_seq: (B, T, D)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)  # Self-attention across T time steps
        return self.proj(attn_out[:, -1])             # Only keep output from last time step


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
        self.temporal_att = TemporalAttention(embed_dim=self.cnn_feature_dim)
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