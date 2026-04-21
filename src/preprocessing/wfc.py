from __future__ import annotations

import pandas as pd


def monthly_attack_country_counts(df_incidents: pd.DataFrame) -> pd.DataFrame:
    x = df_incidents.copy()
    x["month"] = pd.to_datetime(x["date"]).dt.to_period("M").dt.to_timestamp()
    grouped = x.groupby(["month", "attack_type", "country"]).size().rename("count").reset_index()
    pivot = grouped.pivot_table(index="month", columns=["attack_type", "country"], values="count", fill_value=0)
    pivot.columns = [f"NoI_C_{a}__{c}" for a, c in pivot.columns]
    return pivot.reset_index()


def monthly_attack_counts(df_incidents: pd.DataFrame) -> pd.DataFrame:
    x = df_incidents.copy()
    x["month"] = pd.to_datetime(x["date"]).dt.to_period("M").dt.to_timestamp()
    grouped = x.groupby(["month", "attack_type"]).size().rename("count").reset_index()
    pivot = grouped.pivot_table(index="month", columns="attack_type", values="count", fill_value=0)
    pivot.columns = [f"NoI_{c}" for c in pivot.columns]
    return pivot.reset_index()
