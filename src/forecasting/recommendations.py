from __future__ import annotations

import pandas as pd


PRIORITY = {"SWG": 1, "OWG": 2, "ONG": 3, "SNG": 4}
RECOMMEND = {
    "SWG": "Urgent increase in PAT investment and deployment.",
    "OWG": "Increase investment; monitor short-term reversals.",
    "ONG": "Maintain support; gap is narrowing overall.",
    "SNG": "Potentially reallocate resources to lagging PATs.",
}


def build_recommendations(gap_df: pd.DataFrame) -> pd.DataFrame:
    out = gap_df.copy()
    out["priority"] = out["category"].map(PRIORITY)
    out["recommendation"] = out["category"].map(RECOMMEND)
    out = out.sort_values(["priority", "gap_magnitude_2025"], ascending=[True, False])
    return out
