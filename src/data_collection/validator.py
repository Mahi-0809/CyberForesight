"""Post-collection data validation for the monthly dataset.

Checks each column for length, negativity, dead series, and excessive
NaN gaps.  Called at the end of Stage 1 in run_pipeline.py.
"""
from __future__ import annotations

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def validate_monthly_series(
    series: pd.Series,
    name: str,
    expected_months: int = 138,
    min_val: float = 0,
) -> List[str]:
    """Validate a single monthly time-series column.

    Returns a list of human-readable issue strings (empty = OK).
    """
    issues: list[str] = []

    # Check length
    if len(series) != expected_months:
        issues.append(
            f"{name}: expected {expected_months} months, got {len(series)}"
        )

    # Check no negatives
    if (series < min_val).any():
        issues.append(f"{name}: contains values below {min_val}")

    # Check not all zeros (dead series)
    if (series == 0).all():
        issues.append(
            f"{name}: all values are zero -- collection may have failed"
        )

    # Check for large gaps (NaN)
    nan_count = int(series.isna().sum())
    if nan_count > 12:  # more than 1 year of gaps
        issues.append(
            f"{name}: too many missing months ({nan_count})"
        )

    return issues


def validate_all_data(dataset_df: pd.DataFrame) -> List[str]:
    """Validate every numeric column in the assembled dataset.

    Prints a summary of warnings or a success message.
    Returns the full list of issue strings.
    """
    all_issues: list[str] = []
    value_cols = [c for c in dataset_df.columns if c != "month"]

    for col in value_cols:
        issues = validate_monthly_series(dataset_df[col], col)
        all_issues.extend(issues)

    if all_issues:
        print("DATA VALIDATION WARNINGS:")
        for issue in all_issues:
            print(f"  [WARNING] {issue}")
    else:
        print("[OK] All data validation checks passed")

    return all_issues
