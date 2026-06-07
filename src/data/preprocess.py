from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


BAD_STATUSES = [
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Does not meet credit policy. Status: Charged Off",
]

GOOD_STATUSES = [
    "Fully Paid",
    "Does not meet credit policy. Status: Fully Paid",
]

EXCLUDED_STATUSES = [
    "Current",
    "In Grace Period",
    "Late (16-30 days)",
    "Issued",
]


def construct_default_label(df: pd.DataFrame) -> pd.DataFrame:
    """Construct binary default labels from LendingClub loan_status."""
    result = df.copy()
    result["default_label"] = np.nan
    result.loc[result["loan_status"].isin(BAD_STATUSES), "default_label"] = 1
    result.loc[result["loan_status"].isin(GOOD_STATUSES), "default_label"] = 0
    return result


def label_maturity_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Remove loan statuses that do not have mature repayment outcomes."""
    return df[~df["loan_status"].isin(EXCLUDED_STATUSES)].copy()


def construct_sensitive_labels(df: pd.DataFrame, setting: str = "strict") -> pd.DataFrame:
    """Construct labels under supported maturity and delinquency settings."""
    if setting == "strict":
        return construct_default_label(label_maturity_filter(df))

    if setting == "early_delinquency":
        retained_statuses = BAD_STATUSES + GOOD_STATUSES + ["Late (16-30 days)"]
        result = df[df["loan_status"].isin(retained_statuses)].copy()
        result["default_label"] = np.nan
        result.loc[result["loan_status"].isin(BAD_STATUSES + ["Late (16-30 days)"]), "default_label"] = 1
        result.loc[result["loan_status"].isin(GOOD_STATUSES), "default_label"] = 0
        return result

    raise ValueError(f"Unknown label setting: {setting}")


def report_label_distribution(df: pd.DataFrame) -> dict[str, Any]:
    """Report loan_status distribution and exclusion diagnostics."""
    total = len(df)
    status_counts = df["loan_status"].value_counts()
    status_ratios = (status_counts / total).to_dict() if total else {}

    excluded_mask = df["loan_status"].isin(EXCLUDED_STATUSES)
    retained_mask = ~excluded_mask
    excluded_fraction = float(excluded_mask.mean()) if total else 0.0

    compare_cols = [column for column in ["loan_amnt", "dti", "emp_length", "annual_inc"] if column in df.columns]
    distribution_differences = {}
    for column in compare_cols:
        excluded_values = pd.to_numeric(df.loc[excluded_mask, column], errors="coerce").dropna()
        retained_values = pd.to_numeric(df.loc[retained_mask, column], errors="coerce").dropna()
        if len(excluded_values) > 0 and len(retained_values) > 0:
            distribution_differences[column] = {
                "excluded_mean": float(excluded_values.mean()),
                "retained_mean": float(retained_values.mean()),
                "excluded_median": float(excluded_values.median()),
                "retained_median": float(retained_values.median()),
            }

    return {
        "total_samples": total,
        "loan_status_counts": status_counts.to_dict(),
        "loan_status_ratios": status_ratios,
        "excluded_fraction": excluded_fraction,
        "n_excluded": int(excluded_mask.sum()),
        "n_retained": int(retained_mask.sum()),
        "distribution_differences": distribution_differences,
    }
