from __future__ import annotations

import random

from src.model.b_mtgnn import BayesianMTGNN
from src.training.trainer import fit_model


def random_search(search_cfg: dict, fixed_cfg: dict, train_set, val_set, num_nodes: int, tin: int, tout: int, ckpt_dir: str):
    best = {"score": float("inf"), "params": None, "ckpt": None}
    for i in range(search_cfg["random_iterations"]):
        params = {
            "lr": random.uniform(*search_cfg["lr_range"]),
            "dropout": random.uniform(*search_cfg["dropout_range"]),
            "channels": random.choice(search_cfg["conv_channels"]),
            "gcn_depth": random.choice(search_cfg["gcn_depth"]),
        }
        model = BayesianMTGNN(
            num_nodes=num_nodes,
            tin=tin,
            tout=tout,
            channels=params["channels"],
            gcn_depth=params["gcn_depth"],
            dropout=params["dropout"],
            alpha=fixed_cfg["alpha"],
            top_k=fixed_cfg["graph_k"],
            beta=fixed_cfg["beta"],
        )
        ckpt = f"{ckpt_dir}/trial_{i}.pt"
        score = fit_model(model, train_set, val_set, epochs=fixed_cfg["epochs"], lr=params["lr"], checkpoint_path=ckpt)
        if score < best["score"]:
            best = {"score": score, "params": params, "ckpt": ckpt}
        print(f"[search] iter={i+1}/{search_cfg['random_iterations']} best_rse={best['score']:.4f}")
    return best
