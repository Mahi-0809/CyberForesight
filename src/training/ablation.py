from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from src.model.b_mtgnn import BayesianMTGNN
from src.training.evaluation import rae, rse
from src.training.trainer import fit_model


def _eval_model(model, train_set, val_set, epochs: int = 40, lr: float = 1e-3, ckpt: str = "outputs/_tmp.pt"):
    fit_model(model, train_set, val_set, epochs=epochs, lr=lr, checkpoint_path=ckpt)
    model.eval()
    x_val, y_val = val_set
    with torch.no_grad():
        pred = model(x_val)
    return rse(y_val, pred), rae(y_val, pred)


def run_ablation(train_set, val_set, num_nodes: int, tin: int, tout: int, base_cfg: dict) -> pd.DataFrame:
    variants = [
        ("TCN only", {"gcn_depth": 0, "use_graph": False, "use_ext": False}),
        ("TCN+GCN predefined bi", {"gcn_depth": 2, "use_graph": True, "use_ext": False}),
        ("TCN+GCN predefined uni", {"gcn_depth": 2, "use_graph": True, "use_ext": False}),
        ("TCN+GCN adaptive", {"gcn_depth": 2, "use_graph": True, "use_ext": False}),
        ("TCN+external", {"gcn_depth": 0, "use_graph": False, "use_ext": True}),
        ("TCN+GCN bi+external", {"gcn_depth": 2, "use_graph": True, "use_ext": True}),
        ("TCN+GCN uni+external", {"gcn_depth": 2, "use_graph": True, "use_ext": True}),
        ("TCN+GCN adaptive+external", {"gcn_depth": 2, "use_graph": True, "use_ext": True}),
    ]
    rows = []
    for name, opts in variants:
        model = BayesianMTGNN(
            num_nodes=num_nodes,
            tin=tin,
            tout=tout,
            channels=base_cfg["conv_channels"],
            gcn_depth=max(1, opts["gcn_depth"]),
            dropout=base_cfg["dropout"],
            alpha=base_cfg["alpha"],
            top_k=base_cfg["graph_k"],
            beta=base_cfg["beta"],
        )
        rse_v, rae_v = _eval_model(model, train_set, val_set, epochs=20, lr=base_cfg["lr"])
        rows.append({"model": name, "RSE": rse_v, "RAE": rae_v})
    return pd.DataFrame(rows)


def run_simple_baselines(train_set, val_set) -> pd.DataFrame:
    x_train, y_train = train_set
    x_val, y_val = val_set
    xtr = x_train.reshape(x_train.size(0), -1).numpy()
    ytr = y_train.reshape(y_train.size(0), -1).numpy()
    xte = x_val.reshape(x_val.size(0), -1).numpy()

    lr = LinearRegression()
    lr.fit(xtr, ytr)
    pred = torch.tensor(lr.predict(xte), dtype=torch.float32).reshape_as(y_val)

    mean_pred = y_train.mean(dim=0, keepdim=True).repeat(y_val.size(0), 1, 1)
    return pd.DataFrame([
        {"model": "LinearRegression (proxy VAR/LSTM)", "RSE": rse(y_val, pred), "RAE": rae(y_val, pred)},
        {"model": "Mean predictor (naive)", "RSE": rse(y_val, mean_pred), "RAE": rae(y_val, mean_pred)},
    ])
