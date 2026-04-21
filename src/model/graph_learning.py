from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLearningLayer(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int = 16, alpha: float = 1.0, top_k: int = 12):
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        self.e2 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        self.theta1 = nn.Parameter(torch.randn(emb_dim, emb_dim) * 0.1)
        self.theta2 = nn.Parameter(torch.randn(emb_dim, emb_dim) * 0.1)
        self.alpha = alpha
        self.top_k = top_k

    def forward(self) -> torch.Tensor:
        m1 = torch.tanh(self.alpha * (self.e1 @ self.theta1))
        m2 = torch.tanh(self.alpha * (self.e2 @ self.theta2))
        a = F.relu(torch.tanh(self.alpha * (m1 @ m2.T - m2 @ m1.T)))
        if self.top_k < a.size(1):
            vals, idx = torch.topk(a, self.top_k, dim=1)
            mask = torch.zeros_like(a)
            mask.scatter_(1, idx, 1.0)
            a = a * mask
        d = a.sum(dim=1, keepdim=True) + 1e-6
        return a / d
