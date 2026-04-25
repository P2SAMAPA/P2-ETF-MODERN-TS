"""
TimesNet – 2D convolution on time‑frequency representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv3(x))
        out3 = F.relu(self.conv5(x))
        out4 = F.relu(self.conv_pool(self.pool(x)))
        return torch.cat([out1, out2, out3, out4], dim=1)  # (batch, 4*hidden, H, W)

class TimesNet(nn.Module):
    def __init__(self, seq_len, n_vars, hidden_channels, top_k, output_dim):
        super().__init__()
        self.seq_len = seq_len
        self.n_vars = n_vars
        self.top_k = top_k
        self.hidden_channels = hidden_channels
        self.inception = InceptionBlock(1, hidden_channels)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4 * hidden_channels, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, n_vars)
        batch = x.size(0)
        # FFT to find dominant periods
        xf = torch.fft.rfft(x, dim=1)
        amp = xf.abs().mean(dim=2)  # (batch, seq_len//2+1)
        # Get top-k periods (excluding DC)
        _, top_indices = torch.topk(amp[:, 1:], self.top_k, dim=1)
        top_indices = top_indices + 1

        # Use the first dominant period for simplicity
        period = top_indices[0, 0].item()
        period = max(2, min(period, self.seq_len // 2))

        # Pad sequence so that its length is divisible by period
        padded_len = ((self.seq_len + period - 1) // period) * period
        x_pad = F.pad(x, (0, 0, 0, padded_len - self.seq_len))
        # Reshape to 2D: (batch, 1, H, W)
        H = padded_len // period
        W = period
        x_2d = x_pad.reshape(batch, H, W, self.n_vars).permute(0, 3, 1, 2)
        # Average over variables to get single channel
        x_2d = x_2d.mean(dim=1, keepdim=True)  # (batch, 1, H, W)

        out = self.inception(x_2d)              # (batch, 4*hidden, H, W)
        out = self.adaptive_pool(out)           # (batch, 4*hidden, 1, 1)
        out = out.flatten(1)                    # (batch, 4*hidden)
        return self.fc(out)                     # (batch, output_dim)
