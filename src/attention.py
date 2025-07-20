# attention.py
from src.cnn_feature_extractor import *

class SpatialChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()

        # --- Channel Attention ---
        # This MLP learns weights for each channel using global context
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # Pool each channel to a single value (B, C, 1, 1)
            nn.Flatten(1),                  # Flatten to (B, C)
            nn.Linear(in_channels, in_channels // reduction),  # Dim reduction
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),  # Expand back to original size
            nn.Sigmoid()                    # Output weights in [0, 1] for each channel
        )

        # --- Spatial Attention ---
        # Learns weights for spatial positions (i.e., which (h, w) are important)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),  # Kernel sees a 7x7 neighborhood
            nn.Sigmoid()                                # Output shape: (B, 1, H, W)
        )

    def forward(self, x):  # x: (B, C, H, W)
        B, C, H, W = x.size()

        # === Channel Attention ===
        ch_att = self.mlp(x).view(B, C, 1, 1)  # Output: (B, C, 1, 1)
        x = x * ch_att                         # Weight each channel separately

        # === Spatial Attention ===
        avg_out = torch.mean(x, dim=1, keepdim=True)      # (B, 1, H, W) - average over channels
        max_out, _ = torch.max(x, dim=1, keepdim=True)    # (B, 1, H, W) - max over channels
        s_att = self.spatial(torch.cat([avg_out, max_out], dim=1))  # Input: (B, 2, H, W)
        x = x * s_att                                      # Weight each spatial location

        return x  # Output shape: (B, C, H, W)

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
    def __init__(self, observation_space, features_dim=128, temporal_len=3):
        # observation_space.shape: (T, C, H, W) — no batch dimension
        C, H, W = observation_space.shape[1:]
        super().__init__(observation_space, features_dim)

        self.temporal_len = temporal_len  # Not used internally but might help with debugging

        # --- CNN Backbone ---
        # Applies two 2D conv layers on each frame individually
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),     # (C → 32)
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()      # (32 → 64)
        )

        # --- Attention ---
        self.att = SpatialChannelAttention(64)

        # --- Global pooling to remove spatial dimensions (H, W) ---
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (B, 64, 1, 1)

        # --- Temporal attention over frames ---
        self.temporal_att = TemporalAttention(embed_dim=64)

        # --- Final projection to desired feature size (e.g. 128) ---
        self.out = nn.Sequential(
            nn.Linear(64, features_dim),
            nn.Tanh()
        )

    def forward(self, obs):  # obs: (B, T, C, H, W)
        B, T, C, H, W = obs.size()

        # Step 1: flatten time dimension for CNN processing
        obs = obs.view(B * T, C, H, W)         # → (B*T, C, H, W)

        # Step 2: CNN feature extraction on each frame
        x = self.conv(obs)                     # → (B*T, 64, H, W)

        # Step 3: apply attention on each frame
        x = self.att(x)                        # → (B*T, 64, H, W)

        # Step 4: global average pool to reduce spatial dims
        x = self.pool(x).view(B, T, -1)        # → (B, T, 64) — now it's a sequence again

        # Step 5: temporal attention across T frames
        x = self.temporal_att(x)              # → (B, 64)

        # Step 6: project to final feature vector for policy input
        return self.out(x)                    # → (B, features_dim)
