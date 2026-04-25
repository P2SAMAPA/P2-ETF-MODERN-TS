"""
TimesNet – 2D convolution on time‑frequency representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv3(x))
        out3 = self.relu(self.conv5(x))
        out4 = self.relu(self.conv_pool(self.pool(x)))
        return torch.cat([out1, out2, out3, out4], dim=1)

class TimesNet(nn.Module):
    def __init__(self, seq_len, n_vars, hidden_channels, top_k, output_dim):
        super().__init__()
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.top_k = top_k
        self.inception = InceptionBlock(1, hidden_channels)
        self.fc = nn.Linear(hidden_channels * 4 * seq_len, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, n_vars)
        batch = x.size(0)
        # FFT to find dominant periods
        xf = torch.fft.rfft(x, dim=1)  # (batch, seq_len//2+1, n_vars)
        # Take mean amplitude across variables to find global periods
        amp = xf.abs().mean(dim=2)  # (batch, seq_len//2+1)
        # Get top-k periods (excluding DC)
        _, top_indices = torch.topk(amp[:, 1:], self.top_k, dim=1)  # (batch, top_k)
        top_indices = top_indices + 1  # offset by 1

        # For each selected period, reshape into 2D and apply inception
        # Simplified: use the first period
        period = top_indices[0, 0].item()
        period = min(period, self.seq_len // 2)
        # Reshape into (batch, 1, period, seq_len//period) after padding
        padded_len = ((self.seq_len + period - 1) // period) * period
        x_pad = F.pad(x, (0, 0, 0, padded_len - self.seq_len))
        x_2d = x_pad.reshape(batch, padded_len // period, period, self.n_vars).permute(0, 3, 1, 2)
        # Average over variables to get single channel
        x_2d = x_2d.mean(dim=1, keepdim=True)  # (batch, 1, H, W)
        out = self.inception(x_2d)  # (batch, 4*hidden, H, W)
        out = out.flatten(1)
        return self.fc(out)
