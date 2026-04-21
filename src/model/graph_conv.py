from __future__ import annotations

import torch
import torch.nn as nn


class MixHopGraphConv(nn.Module):
    def __init__(self, channels: int, gcn_depth: int = 2, beta: float = 0.2):
        super().__init__()
        self.gcn_depth = gcn_depth
        self.beta = beta
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(channels, channels) * 0.05) for _ in range(gcn_depth + 1)])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N, T]
        h = x
        outs = [torch.einsum("bcnt,cd->bdnt", h, self.weights[0])]
        h_prev = h
        for k in range(1, self.gcn_depth + 1):
            prop = torch.einsum("nm,bcmt->bcnt", adj, h_prev)
            h_prev = self.beta * h + (1 - self.beta) * prop
            outs.append(torch.einsum("bcnt,cd->bdnt", h_prev, self.weights[k]))
        return sum(outs)
