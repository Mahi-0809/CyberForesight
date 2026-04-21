from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.constants import PAT_CODES, THREATS, THREAT_PAT_MAP


def compute_gap_series(df: pd.DataFrame, threat: str, pat: str) -> pd.Series:
    noi_col = f"NoI_{threat}"
    nom_col = f"NoM_P_{pat}"
    noi = df[noi_col] / max(df[noi_col].max(), 1e-6)
    nom = df[nom_col] / max(df[nom_col].max(), 1e-6)
    return noi - nom


def build_tpt_graph(df: pd.DataFrame) -> dict:
    nodes = [{"id": t, "type": "threat"} for t in THREATS] + [{"id": p, "type": "pat"} for p in PAT_CODES]
    nodes += [{"id": "ACA", "type": "context"}, {"id": "PH", "type": "context"}]
    edges = []
    for threat, pats in THREAT_PAT_MAP.items():
        for pat in pats:
            gap = compute_gap_series(df, threat, pat)
            edges.append({
                "source": threat,
                "target": pat,
                "weights": [{"month": m.strftime("%Y-%m"), "gap": float(v)} for m, v in zip(df["month"], gap)],
            })
    return {"nodes": nodes, "edges": edges}


def save_graph(graph: dict, root_dir: str):
    out = Path(root_dir) / "data" / "graph"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tpt_graph.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
