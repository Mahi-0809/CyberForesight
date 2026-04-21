from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.constants import PAT_CODES, THREATS, THREAT_TYPE
from src.data_collection.validator import validate_all_data
from src.forecasting.forecast import forecast_with_ci
from src.forecasting.gap_analysis import compute_gap_report
from src.forecasting.recommendations import build_recommendations
from src.graph.adjacency import node_list
from src.graph.tpt_graph import build_tpt_graph, save_graph
from src.model.b_mtgnn import BayesianMTGNN
from src.preprocessing.dataset_builder import build_monthly_dataset, save_dataset
from src.training.hyperparam_search import random_search
from src.training.trainer import make_windows, split_data
from src.training.ablation import run_ablation, run_simple_baselines
from src.training.evaluation import all_metrics
from src.utils import ensure_dir, load_config, set_seed
from src.visualisation.atc_plot import plot_atc
from src.visualisation.trend_plots import plot_threat_forecasts


def _align_node_matrix(df: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    x = pd.DataFrame({"month": df["month"]})
    for n in nodes:
        if n in THREATS and f"NoI_{n}" in df.columns:
            x[n] = df[f"NoI_{n}"]
        elif n in PAT_CODES and f"NoM_P_{n}" in df.columns:
            x[n] = df[f"NoM_P_{n}"]
        elif n in ("ACA", "PH") and n in df.columns:
            x[n] = df[n]
        else:
            x[n] = 0.0
    return x


def _build_atc_phases(forecast_df: pd.DataFrame) -> pd.DataFrame:
    pats = forecast_df[forecast_df["node"].isin(PAT_CODES)]
    rows = []
    for i, pat in enumerate(sorted(PAT_CODES)):
        p = pats[pats["node"] == pat].sort_values("month")
        if p.empty:
            continue
        vals = p["pred"].values
        slope = np.polyfit(np.arange(min(len(vals), 12)), vals[: min(len(vals), 12)], 1)[0]
        if slope > 0.2:
            phase, x, y = "Growth", 3 + (i % 10) * 0.15, 2.5 + (i // 10) * 0.1
        elif slope > -0.05:
            phase, x, y = "Maturity/Stability", 7 + (i % 10) * 0.12, 2.0 + (i // 10) * 0.09
        else:
            phase, x, y = "Trough", 5 + (i % 10) * 0.12, 1.3 + (i // 10) * 0.08
        color = "blue" if slope > 0.05 else ("purple" if slope < -0.05 else "grey")
        rows.append({"pat": pat, "phase": phase, "slope": slope, "color": color, "x": x, "y": y, "relevance": "Both"})
    return pd.DataFrame(rows)


def _recommendations_pdf(df: pd.DataFrame, path: str):
    top = df.head(20)[["threat", "pat", "gap_2023", "gap_2024", "gap_2025", "category"]]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    tbl = ax.table(cellText=top.round(3).values, colLabels=top.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full B-MTGNN cyber-threat forecasting pipeline."
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        default=False,
        help="Force re-download of all cached API data.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).parent
    cfg = load_config(str(root / "config.yaml"))
    set_seed(cfg["project"]["seed"])

    # Apply CLI overrides
    if args.refresh_cache:
        cfg.setdefault("cache", {})["refresh"] = True
        print("[REFRESH] Cache refresh enabled -- all data will be re-downloaded")

    use_live = cfg.get("api", {}).get("use_live_apis", False)
    if use_live:
        print("[LIVE] Live API mode ENABLED -- will attempt real data collection")
    else:
        print("[SYNTHETIC] Synthetic mode -- using generated data (set use_live_apis: true for real data)")

    print("\n[Stage 1] Building monthly dataset...")
    df = build_monthly_dataset(cfg)
    save_dataset(df, str(root))
    print(f"  dataset shape: {df.shape}")
    print(f"  month range: {df['month'].min()} to {df['month'].max()}")

    # Validate collected data
    print("\n[Validation] Checking data quality...")
    issues = validate_all_data(df)
    if issues:
        print(f"  Found {len(issues)} warnings (non-fatal -- pipeline will continue)")

    print("\n[Stage 2] Building TPT graph...")
    graph = build_tpt_graph(df)
    save_graph(graph, str(root))
    print("  nodes:", len(graph["nodes"]), "edges:", len(graph["edges"]))

    print("\n[Stage 3] Training B-MTGNN...")
    nodes = node_list()
    node_df = _align_node_matrix(df, nodes)
    values = node_df[nodes].values.astype(np.float32)
    tin, tout = cfg["model"]["tin"], cfg["model"]["tout"]
    x, y = make_windows(values, tin, tout)
    train_set, val_set, _ = split_data(x, y)
    ensure_dir(str(root / "outputs"))
    best = random_search(cfg["search"], cfg["model"], train_set, val_set, len(nodes), tin, tout, str(root / "outputs"))
    print("  best params:", best["params"], "best val RSE:", round(best["score"], 4))

    print("\n[Ablation] Running model variants and baselines...")
    ablation_df = run_ablation(train_set, val_set, len(nodes), tin, tout, cfg["model"])
    baseline_df = run_simple_baselines(train_set, val_set)
    pd.concat([ablation_df, baseline_df], ignore_index=True).to_csv(root / "outputs" / "ablation_and_baselines.csv", index=False)

    model = BayesianMTGNN(
        num_nodes=len(nodes),
        tin=tin,
        tout=tout,
        channels=best["params"]["channels"],
        gcn_depth=best["params"]["gcn_depth"],
        dropout=best["params"]["dropout"],
        alpha=cfg["model"]["alpha"],
        top_k=cfg["model"]["graph_k"],
        beta=cfg["model"]["beta"],
    )
    model.load_state_dict(torch.load(best["ckpt"], map_location="cpu"))
    torch.save(model.state_dict(), root / "outputs" / "b_mtgnn_best.pt")

    model.eval()
    x_val, y_val = val_set
    with torch.no_grad():
        val_pred = model(x_val)
    m = all_metrics(y_val, val_pred)
    pd.DataFrame([m]).to_csv(root / "outputs" / "validation_metrics.csv", index=False)
    print(
        "  validation metrics (best checkpoint on val):",
        f"RSE={m['RSE']:.4f} RAE={m['RAE']:.4f} RMSE={m['RMSE']:.4f} R2={m['R2']:.4f}",
    )

    print("\n[Stage 4] Forecasting and gap analysis...")
    x_last = x[-1:].clone()
    months = pd.date_range(cfg["project"]["forecast_start"], cfg["project"]["forecast_end"], freq="MS")
    fc = forecast_with_ci(model, x_last, nodes, months, mc_it=cfg["model"]["mc_iterations"])
    fc.to_csv(root / "outputs" / "forecast_2023_2025.csv", index=False)

    gap_df = compute_gap_report(fc)
    gap_df.to_csv(root / "outputs" / "gap_analysis_report.csv", index=False)
    rec_df = build_recommendations(gap_df)
    _recommendations_pdf(rec_df, str(root / "outputs" / "investment_recommendations.pdf"))
    plot_threat_forecasts(fc, THREATS, str(root / "outputs" / "trend_plots"))
    print("  forecast rows:", len(fc), "gap rows:", len(gap_df))

    print("\n[Stage 5] ATC analysis...")
    atc_df = _build_atc_phases(fc)
    atc_df.to_csv(root / "outputs" / "atc_phases.csv", index=False)
    plot_atc(atc_df, str(root / "outputs" / "atc_diagram.png"))
    print("  atc entries:", len(atc_df))
    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
