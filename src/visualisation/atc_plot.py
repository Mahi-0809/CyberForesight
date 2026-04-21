from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_atc(pat_phase_df: pd.DataFrame, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0, 10, 500)
    y = np.piecewise(
        x,
        [x < 2, (x >= 2) & (x < 4), (x >= 4) & (x < 6), (x >= 6) & (x < 8), x >= 8],
        [lambda t: 0.2 * t + 0.5, lambda t: 1.2 * (t - 2) + 0.9, lambda t: -0.6 * (t - 4) + 3.3, lambda t: 0.5 * (t - 6) + 2.1, lambda t: 0.1 * (t - 8) + 3.1],
    )
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, color="black", linewidth=2, label="ATC Curve")
    for _, row in pat_phase_df.iterrows():
        marker = {"RI": "o", "E": "s", "Both": "^"}.get(row["relevance"], "^")
        plt.scatter(row["x"], row["y"], c=row["color"], marker=marker, s=45)
        plt.text(row["x"] + 0.03, row["y"] + 0.03, row["pat"], fontsize=7)
    plt.title("Alleviation Technologies Cycle (ATC)")
    plt.xlabel("Lifecycle progression")
    plt.ylabel("Visibility")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
