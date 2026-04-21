from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.graph_conv import MixHopGraphConv
from src.model.graph_learning import GraphLearningLayer
from src.model.temporal_conv import DilatedInception


class MTGNN(nn.Module):
    def __init__(self, num_nodes: int, tin: int, tout: int, channels: int = 16, gcn_depth: int = 2, dropout: float = 0.4, alpha: float = 1.5, top_k: int = 12, beta: float = 0.2):
        super().__init__()
        self.tout = tout
        self.input_proj = nn.Conv2d(1, channels, kernel_size=(1, 1))
        self.temporal = DilatedInception(channels, channels, dilation=1)
        self.graph_learner = GraphLearningLayer(num_nodes, emb_dim=16, alpha=alpha, top_k=top_k)
        self.graph_conv = MixHopGraphConv(channels, gcn_depth=gcn_depth, beta=beta)
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(channels, tout, kernel_size=(1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Tin, N, 1] -> [B, 1, N, Tin]
        x = x.permute(0, 3, 2, 1)
        h = self.input_proj(x)
        h = F.relu(self.temporal(h))
        adj = self.graph_learner()
        h = F.relu(self.graph_conv(h, adj))
        h = self.dropout(h)
        y = self.readout(h).mean(dim=-1)  # [B, Tout, N]
        return y
