import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    BAD_STATUSES,
    EXCLUDED_STATUSES,
    GOOD_STATUSES,
    construct_default_label,
    construct_sensitive_labels,
    label_maturity_filter,
    report_label_distribution,
)


def test_charged_off_is_bad():
    df = pd.DataFrame({"loan_status": ["Charged Off", "Fully Paid"]})

    result = construct_default_label(df)

    assert result["default_label"].tolist() == [1, 0]


def test_fully_paid_is_good():
    df = pd.DataFrame({"loan_status": ["Fully Paid", "Fully Paid"]})

    result = construct_default_label(df)

    assert result["default_label"].tolist() == [0, 0]


def test_current_is_excluded():
    df = pd.DataFrame({"loan_status": ["Current", "Fully Paid", "Charged Off"]})

    result = label_maturity_filter(df)

    assert "Current" not in result["loan_status"].values


def test_in_grace_period_is_excluded():
    df = pd.DataFrame({"loan_status": ["In Grace Period", "Fully Paid"]})

    result = label_maturity_filter(df)

    assert len(result) == 1


def test_default_label_values():
    df = pd.DataFrame(
        {
            "loan_status": [
                "Charged Off",
                "Default",
                "Late (31-120 days)",
                "Does not meet credit policy. Status: Charged Off",
                "Fully Paid",
                "Does not meet credit policy. Status: Fully Paid",
            ]
        }
    )

    result = construct_default_label(df)

    assert result["default_label"].tolist()[:4] == [1, 1, 1, 1]
    assert result["default_label"].tolist()[4:] == [0, 0]


def test_unknown_status_keeps_missing_label():
    df = pd.DataFrame({"loan_status": ["Unknown Status"]})

    result = construct_default_label(df)

    assert np.isnan(result.loc[0, "default_label"])


def test_strict_sensitive_labels_filter_excluded_statuses():
    df = pd.DataFrame({"loan_status": ["Late (16-30 days)", "Current", "Fully Paid"]})

    result = construct_sensitive_labels(df, setting="strict")

    assert result["loan_status"].tolist() == ["Fully Paid"]
    assert result["default_label"].tolist() == [0]


def test_early_delinquency_sensitive_labels_treat_late_16_30_as_bad():
    df = pd.DataFrame({"loan_status": ["Late (16-30 days)", "Current", "Fully Paid"]})

    result = construct_sensitive_labels(df, setting="early_delinquency")

    assert result["loan_status"].tolist() == ["Late (16-30 days)", "Fully Paid"]
    assert result["default_label"].tolist() == [1, 0]


def test_unknown_sensitive_label_setting_raises():
    df = pd.DataFrame({"loan_status": ["Fully Paid"]})

    with pytest.raises(ValueError, match="Unknown label setting"):
        construct_sensitive_labels(df, setting="not_a_setting")


def test_report_label_distribution_counts_exclusions_and_numeric_differences():
    df = pd.DataFrame(
        {
            "loan_status": ["Current", "In Grace Period", "Fully Paid", "Charged Off"],
            "loan_amnt": [10000, 20000, 30000, 40000],
            "dti": [10.0, 20.0, 30.0, 40.0],
            "emp_length": [1, 2, 3, 4],
            "annual_inc": [50000, 60000, 70000, 80000],
        }
    )

    report = report_label_distribution(df)

    assert report["total_samples"] == 4
    assert report["n_excluded"] == 2
    assert report["n_retained"] == 2
    assert report["excluded_fraction"] == 0.5
    assert report["loan_status_counts"]["Current"] == 1
    assert report["loan_status_ratios"]["Fully Paid"] == 0.25
    assert report["distribution_differences"]["loan_amnt"] == {
        "excluded_mean": 15000.0,
        "retained_mean": 35000.0,
        "excluded_median": 15000.0,
        "retained_median": 35000.0,
    }


def test_status_constant_sets_match_spec():
    assert "Charged Off" in BAD_STATUSES
    assert "Fully Paid" in GOOD_STATUSES
    assert "Current" in EXCLUDED_STATUSES
