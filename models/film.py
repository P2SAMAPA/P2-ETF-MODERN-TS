"""
FiLM – Feature‑wise Linear Modulation with macro conditioning.
"""

import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.gamma_generator = nn.Linear(cond_dim, feature_dim)
        self.beta_generator = nn.Linear(cond_dim, feature_dim)

    def forward(self, x, cond):
        gamma = self.gamma_generator(cond)
        beta = self.beta_generator(cond)
        return gamma * x + beta

class FiLM(nn.Module):
    def __init__(self, seq_len, n_vars, cond_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(n_vars, hidden_dim)
        self.film1 = FiLMLayer(hidden_dim, cond_dim)
        self.film2 = FiLMLayer(hidden_dim, cond_dim)
        self.head = nn.Linear(hidden_dim * seq_len, output_dim)

    def forward(self, x, cond):
        # x: (batch, seq_len, n_vars), cond: (batch, cond_dim)
        batch, seq_len, n_vars = x.shape
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        x = self.film1(x, cond.unsqueeze(1).expand(-1, seq_len, -1))
        x = torch.relu(x)
        x = self.film2(x, cond.unsqueeze(1).expand(-1, seq_len, -1))
        x = x.flatten(1)  # (batch, seq_len*hidden_dim)
        return self.head(x)
