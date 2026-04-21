from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_threat_forecasts(df_forecast: pd.DataFrame, threats: list[str], out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for threat in threats:
        x = df_forecast[df_forecast["node"] == threat].copy()
        if x.empty:
            continue
        x = x.sort_values("month")
        plt.figure(figsize=(10, 4))
        plt.plot(x["month"], x["pred"], label="Forecast")
        plt.fill_between(x["month"], x["ci_lower"], x["ci_upper"], alpha=0.2, label="95% CI")
        plt.title(threat)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        plt.savefig(Path(out_dir) / f"{threat.replace(' ', '_')}.png", dpi=150)
        plt.close()
