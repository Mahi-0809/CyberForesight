from __future__ import annotations

import pandas as pd

from src.constants import THREAT_PAT_MAP


def _category(g23: float, g24: float, g25: float) -> str:
    if g23 < g24 < g25:
        return "SWG"
    if g23 > g24 > g25:
        return "SNG"
    if g25 > g23:
        return "OWG"
    return "ONG"


def compute_gap_report(df_forecast: pd.DataFrame) -> pd.DataFrame:
    pivot = df_forecast.pivot_table(index="month", columns="node", values="pred", aggfunc="mean")
    out = []
    for t, pats in THREAT_PAT_MAP.items():
        if t not in pivot.columns:
            continue
        t_norm = pivot[t] / max(pivot[t].max(), 1e-6)
        for p in pats:
            if p not in pivot.columns:
                continue
            p_norm = pivot[p] / max(pivot[p].max(), 1e-6)
            gap = t_norm - p_norm
            y = gap.groupby(gap.index.year).mean()
            g23, g24, g25 = float(y.get(2023, 0.0)), float(y.get(2024, 0.0)), float(y.get(2025, 0.0))
            out.append({
                "threat": t,
                "pat": p,
                "gap_2023": g23,
                "gap_2024": g24,
                "gap_2025": g25,
                "category": _category(g23, g24, g25),
                "gap_magnitude_2025": abs(g25),
            })
    return pd.DataFrame(out).sort_values(["category", "gap_magnitude_2025"], ascending=[True, False])
