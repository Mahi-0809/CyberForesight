from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.training.evaluation import rae, rse


def make_windows(values: np.ndarray, tin: int, tout: int):
    x, y = [], []
    for i in range(0, len(values) - tin - tout + 1):
        x.append(values[i : i + tin])
        y.append(values[i + tin : i + tin + tout])
    x = np.stack(x, axis=0)[..., None]  # [B, Tin, N, 1]
    y = np.stack(y, axis=0)  # [B, Tout, N]
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def split_data(x: torch.Tensor, y: torch.Tensor, train_ratio: float = 0.43, val_ratio: float = 0.30):
    n = len(x)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (x[:n_train], y[:n_train]), (x[n_train:n_train + n_val], y[n_train:n_train + n_val]), (x[n_train + n_val:], y[n_train + n_val:])


def fit_model(model, train_set, val_set, epochs: int, lr: float, checkpoint_path: str):
    x_train, y_train = train_set
    x_val, y_val = val_set
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    best_rse = float("inf")
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                vpred = model(x_val)
                v_rse = rse(y_val, vpred)
                v_rae = rae(y_val, vpred)
            print(f"epoch={ep:03d} train_mae={loss.item():.4f} val_rse={v_rse:.4f} val_rae={v_rae:.4f}")
            if v_rse < best_rse:
                best_rse = v_rse
                torch.save(model.state_dict(), checkpoint_path)
    return best_rse
