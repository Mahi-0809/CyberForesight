from __future__ import annotations

import pandas as pd


def top_gap_table(gap_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return gap_df.sort_values("gap_magnitude_2025", ascending=False).head(n)
