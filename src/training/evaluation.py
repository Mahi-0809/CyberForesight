from __future__ import annotations

import torch


def rse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    num = torch.sqrt(torch.sum((y_true - y_pred) ** 2))
    den = torch.sqrt(torch.sum((y_true - y_true.mean()) ** 2) + 1e-8)
    return (num / den).item()


def rae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    num = torch.sum(torch.abs(y_true - y_pred))
    den = torch.sum(torch.abs(y_true - y_true.mean()) + 1e-8)
    return (num / den).item()


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Root mean squared error (same units as targets)."""
    mse = torch.mean((y_true - y_pred) ** 2)
    return torch.sqrt(mse + 1e-12).item()


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Coefficient of determination; 1.0 is perfect, can be negative if worse than mean."""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2) + 1e-8
    return (1.0 - ss_res / ss_tot).item()


def all_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    return {
        "RSE": rse(y_true, y_pred),
        "RAE": rae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
