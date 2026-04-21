from __future__ import annotations

import holidays
import pandas as pd


def monthly_holiday_counts(countries: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    months = pd.date_range(start_date, end_date, freq="MS")
    rows = []
    for month in months:
        row = {"month": month}
        for code in countries:
            try:
                cal = holidays.country_holidays(code, years=month.year)
                count = sum(1 for d in cal.keys() if d.year == month.year and d.month == month.month)
            except Exception:
                count = 1
            row[f"PH_C_{code}"] = count
        rows.append(row)
    df = pd.DataFrame(rows)
    df["PH"] = df[[c for c in df.columns if c.startswith("PH_C_")]].sum(axis=1)
    return df
