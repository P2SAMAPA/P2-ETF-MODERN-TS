"""
TSMixer – All‑MLP architecture for time series.
"""

import torch
import torch.nn as nn

class TSMixerBlock(nn.Module):
    def __init__(self, seq_len, n_vars, hidden_dim):
        super().__init__()
        self.time_mix = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len)
        )
        self.feature_mix = nn.Sequential(
            nn.Linear(n_vars, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_vars)
        )
        self.norm1 = nn.LayerNorm(n_vars)
        self.norm2 = nn.LayerNorm(n_vars)

    def forward(self, x):
        # x: (batch, seq_len, n_vars)
        # Time mixing
        residual = x
        x_t = x.permute(0, 2, 1)  # (batch, n_vars, seq_len)
        x_t = self.time_mix(x_t).permute(0, 2, 1)  # back to (batch, seq_len, n_vars)
        x = self.norm1(residual + x_t)
        # Feature mixing
        residual = x
        x_f = self.feature_mix(x)
        x = self.norm2(residual + x_f)
        return x

class TSMixer(nn.Module):
    def __init__(self, seq_len, n_vars, hidden_dim, num_blocks, output_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            TSMixerBlock(seq_len, n_vars, hidden_dim) for _ in range(num_blocks)
        ])
        self.head = nn.Linear(n_vars, output_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        # Take last time step
        x = x[:, -1, :]  # (batch, n_vars)
        return self.head(x)
