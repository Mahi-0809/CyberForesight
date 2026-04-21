from __future__ import annotations

import pandas as pd
import torch


def forecast_with_ci(model, x_last: torch.Tensor, node_names: list[str], months: pd.DatetimeIndex, mc_it: int = 30) -> pd.DataFrame:
    mean_pred, std_pred, lo, hi = model.predict_with_uncertainty(x_last, it=mc_it)
    # mean_pred: [1, Tout, N]
    mean_np = mean_pred.squeeze(0).cpu().numpy()
    lo_np = lo.squeeze(0).cpu().numpy()
    hi_np = hi.squeeze(0).cpu().numpy()
    rows = []
    for t_idx, month in enumerate(months):
        for n_idx, node in enumerate(node_names):
            rows.append({
                "month": month,
                "node": node,
                "pred": float(mean_np[t_idx, n_idx]),
                "ci_lower": float(lo_np[t_idx, n_idx]),
                "ci_upper": float(hi_np[t_idx, n_idx]),
            })
    return pd.DataFrame(rows)
