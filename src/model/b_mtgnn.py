from __future__ import annotations

import torch
import torch.nn as nn

from src.model.mtgnn import MTGNN


class BayesianMTGNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.base = MTGNN(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, it: int = 30):
        self.train()  # keep dropout active for MC dropout
        preds = [self.forward(x) for _ in range(it)]
        stack = torch.stack(preds, dim=0)
        mean_pred = stack.mean(dim=0)
        std_pred = stack.std(dim=0)
        return mean_pred, std_pred, mean_pred - 1.96 * std_pred, mean_pred + 1.96 * std_pred
