"""
PatchTST – Patch‑based Transformer for time series.
"""

import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, embed_dim):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, n_vars)
        # For simplicity, we treat each variable independently and share weights.
        # Here we average over variables to get a single channel.
        x = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        x = x.squeeze(-1)                 # (batch, seq_len)
        # Create patches: unfold
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # (batch, n_patches, patch_len)
        x = self.proj(x)  # (batch, n_patches, embed_dim)
        return x

class PatchTST(nn.Module):
    def __init__(self, n_vars, seq_len, patch_len, stride, embed_dim, num_heads, num_layers, output_dim):
        super().__init__()
        self.seq_len = seq_len
        self.patch_embed = PatchEmbedding(patch_len, stride, embed_dim)
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        batch = x.size(0)
        patches = self.patch_embed(x)  # (batch, n_patches, embed_dim)
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, patches], dim=1)   # (batch, n_patches+1, embed_dim)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.head(x[:, 0, :])  # use CLS token
        return x
