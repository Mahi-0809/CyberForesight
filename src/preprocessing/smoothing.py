from __future__ import annotations

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def apply_des(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in value_columns:
        series = out[col].astype(float)
        try:
            model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
            fit = model.fit(optimized=True, use_brute=False)
            out[col] = fit.fittedvalues
        except Exception:
            out[col] = series.ewm(alpha=0.2, adjust=False).mean()
    return out
