"""Build the (138 × 1160) monthly dataset from all data collectors.

This module orchestrates the four data sources:
  - NoI  (Hackmageddon / NVD) → incident counts per attack type / country
  - NoM_A (Elsevier / Semantic Scholar) → academic mentions per threat
  - NoM_P (Elsevier / Semantic Scholar) → academic mentions per PAT
  - ACA  (GDELT / ACLED) → armed-conflict events per country
  - PH   (holidays library) → public holidays per country  [unchanged]

Each collector receives the full *config* dict so it can read API keys,
rate limits, and cache settings autonomously.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import COUNTRIES_36, PAT_CODES, THREATS
from src.data_collection.elsevier_api import ElsevierCollector
from src.data_collection.hackmageddon import HackmageddonCollector
from src.data_collection.holidays import monthly_holiday_counts
from src.data_collection.twitter_api import TwitterConflictCollector
from src.preprocessing.smoothing import apply_des
from src.preprocessing.wfc import monthly_attack_country_counts, monthly_attack_counts


def build_monthly_dataset(config: dict) -> pd.DataFrame:
    p = config["project"]
    use_live = config.get("api", {}).get("use_live_apis", False)
    start, end = p["start_date"], p["end_date"]

    # ---- NoI: Number of Incidents ----
    print("  Collecting incident data (NoI)...")
    hm = HackmageddonCollector(use_live_api=use_live, config=config)
    incidents = hm.impute_missing_country(hm.collect(start, end))
    noi_c = monthly_attack_country_counts(incidents)
    noi = monthly_attack_counts(incidents)

    # ---- NoM_A: Academic mentions of attack types ----
    print(f"  Collecting academic mentions for {len(THREATS)} threats (NoM_A)...")
    elsevier = ElsevierCollector(
        api_key=config.get("api", {}).get("elsevier_key", ""),
        use_live_api=use_live,
        config=config,
    )
    nom_a = elsevier.monthly_mentions(THREATS, start, end, seed=42).rename(
        columns={t: f"NoM_A_{t}" for t in THREATS}
    )

    # ---- NoM_P: Academic mentions of PATs ----
    print(f"  Collecting academic mentions for {len(PAT_CODES)} PATs (NoM_P)...")
    nom_p = elsevier.monthly_mentions(PAT_CODES, start, end, seed=84).rename(
        columns={p_code: f"NoM_P_{p_code}" for p_code in PAT_CODES}
    )

    # ---- ACA: Armed Conflict Area events ----
    print("  Collecting armed-conflict data (ACA)...")
    tw = TwitterConflictCollector(
        bearer_token=config.get("api", {}).get("twitter_bearer", ""),
        use_live_api=use_live,
        config=config,
    )
    aca = tw.monthly_conflict_counts(COUNTRIES_36, start, end)

    # ---- PH: Public Holidays (unchanged) ----
    print("  Collecting public-holiday data (PH)...")
    ph = monthly_holiday_counts(COUNTRIES_36, start, end)

    # ---- Merge everything on month ----
    months = pd.DataFrame({"month": pd.date_range(start, end, freq="MS")})
    data = months.merge(noi_c, on="month", how="left").merge(noi, on="month", how="left")
    data = data.merge(nom_a, on="month", how="left").merge(nom_p, on="month", how="left")
    data = data.merge(aca, on="month", how="left").merge(ph, on="month", how="left")
    data = data.fillna(0).sort_values("month").reset_index(drop=True)

    # ---- Ensure ALL expected columns exist (real data may lack some) ----
    # NoI per-country: NoI_C_{threat}__{country}
    for t in THREATS:
        for c in COUNTRIES_36:
            col = f"NoI_C_{t}__{c}"
            if col not in data.columns:
                data[col] = 0.0
    # NoI aggregate: NoI_{threat}
    for t in THREATS:
        col = f"NoI_{t}"
        if col not in data.columns:
            data[col] = 0.0
    # NoM_A: NoM_A_{threat}
    for t in THREATS:
        col = f"NoM_A_{t}"
        if col not in data.columns:
            data[col] = 0.0
    # NoM_P: NoM_P_{pat}
    for p_code in PAT_CODES:
        col = f"NoM_P_{p_code}"
        if col not in data.columns:
            data[col] = 0.0
    # ACA per-country: ACA_C_{country}
    for c in COUNTRIES_36:
        col = f"ACA_C_{c}"
        if col not in data.columns:
            data[col] = 0.0
    # ACA aggregate
    if "ACA" not in data.columns:
        data["ACA"] = 0.0

    value_cols = [c for c in data.columns if c != "month"]
    smoothed = apply_des(data, value_cols)
    return smoothed


def save_dataset(df: pd.DataFrame, root_dir: str):
    out = Path(root_dir) / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "monthly_dataset.csv", index=False)
