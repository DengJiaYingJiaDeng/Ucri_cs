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

SHARED_FEATURE_MAPPING = {
    "loan_amnt": "loan_amount",
    "dti": "dti",
    "addr_state": "state",
    "emp_length": "emp_length",
    "purpose": "loan_purpose",
    "home_ownership": "home_ownership",
    "annual_inc": "annual_inc",
    "verification_status": "verification_status",
    "delinq_2yrs": "delinq_2yrs",
    "open_acc": "open_acc",
    "revol_bal": "revol_bal",
    "revol_util": "revol_util",
    "total_acc": "total_acc",
    "issue_d": "application_date",
    "term": "term",
    "fico_range_low": "fico_range_low",
    "fico_range_high": "fico_range_high",
    "zip_code": "zip3",
}

REJECTED_FEATURE_MAPPING = {
    "Amount Requested": "loan_amount",
    "Debt-To-Income Ratio": "dti",
    "State": "state",
    "Employment Length": "emp_length",
    "Risk_Score": "risk_score",
    "Application Date": "application_date",
    "Loan Title": "loan_purpose",
    "Zip Code": "zip3",
    "Policy Code": "policy_code",
}

RISK_SCORE_SETTINGS = ["no_riskscore", "input_riskscore", "anchor_riskscore"]


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


def build_shared_features(
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    risk_score_setting: str = "no_riskscore",
) -> pd.DataFrame:
    """Build a shared feature view aligned across accepted and rejected records."""
    if risk_score_setting not in RISK_SCORE_SETTINGS:
        raise ValueError(
            f"Unknown risk_score_setting: {risk_score_setting}. "
            f"Expected one of {RISK_SCORE_SETTINGS}."
        )

    accepted_shared = pd.DataFrame(index=accepted.index)
    for source_column, target_column in SHARED_FEATURE_MAPPING.items():
        if source_column in accepted.columns:
            accepted_shared[target_column] = accepted[source_column]
    accepted_shared["source"] = "accepted"
    accepted_shared["accepted_indicator"] = 1

    rejected_shared = pd.DataFrame(index=rejected.index)
    for source_column, target_column in REJECTED_FEATURE_MAPPING.items():
        if source_column in rejected.columns:
            rejected_shared[target_column] = rejected[source_column]
    rejected_shared["source"] = "rejected"
    rejected_shared["accepted_indicator"] = 0

    combined = pd.concat([accepted_shared, rejected_shared], ignore_index=True)

    if risk_score_setting == "no_riskscore":
        combined = combined.drop(columns=["risk_score"], errors="ignore")
    elif risk_score_setting == "anchor_riskscore":
        combined["risk_score_anchor"] = combined["risk_score"]
        combined = combined.drop(columns=["risk_score"], errors="ignore")

    return combined


def build_accepted_rich_features(accepted: pd.DataFrame) -> pd.DataFrame:
    """Build the accepted-rich feature view for accepted-only PD baselines."""
    result = accepted.copy()
    if {"fico_range_low", "fico_range_high"}.issubset(result.columns):
        result["fico_avg"] = (result["fico_range_low"] + result["fico_range_high"]) / 2

    rich_columns = [
        "loan_amnt",
        "dti",
        "emp_length",
        "addr_state",
        "purpose",
        "home_ownership",
        "annual_inc",
        "verification_status",
        "delinq_2yrs",
        "fico_avg",
        "open_acc",
        "revol_bal",
        "revol_util",
        "total_acc",
        "issue_d",
        "term",
        "int_rate",
        "installment",
        "grade",
        "sub_grade",
    ]
    return result[[column for column in rich_columns if column in result.columns]]
