import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    BAD_STATUSES,
    EXCLUDED_STATUSES,
    GOOD_STATUSES,
    build_accepted_rich_features,
    build_shared_features,
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


def test_build_shared_features():
    accepted = pd.DataFrame(
        {
            "loan_amnt": [10000, 20000],
            "dti": [15.0, 22.0],
            "emp_length": [5, 2],
            "addr_state": ["CA", "TX"],
            "purpose": ["debt_consolidation", "home_improvement"],
            "home_ownership": ["RENT", "MORTGAGE"],
            "annual_inc": [60000, 80000],
            "verification_status": ["Verified", "Not Verified"],
            "delinq_2yrs": [0, 1],
            "fico_range_low": [680, 640],
            "fico_range_high": [684, 644],
            "open_acc": [10, 15],
            "revol_bal": [15000, 25000],
            "revol_util": [45.0, 60.0],
            "total_acc": [20, 30],
            "issue_d": ["2013-06", "2014-01"],
            "term": [36, 60],
            "zip_code": ["945xx", "750xx"],
            "int_rate": [10.5, 14.0],
            "grade": ["B", "C"],
        }
    )
    rejected = pd.DataFrame(
        {
            "Amount Requested": [15000, 25000],
            "Debt-To-Income Ratio": [18.5, 28.0],
            "State": ["CA", "TX"],
            "Employment Length": [3, 1],
            "Risk_Score": [680, 590],
            "Application Date": ["2013-07", "2014-03"],
            "Loan Title": ["debt_consolidation", "business"],
            "Zip Code": ["945xx", "750xx"],
            "Policy Code": [0, 0],
        }
    )

    result = build_shared_features(accepted, rejected)

    assert "loan_amount" in result.columns
    assert "dti" in result.columns
    assert "state" in result.columns
    assert "emp_length" in result.columns
    assert "loan_purpose" in result.columns
    assert "zip3" in result.columns
    assert "risk_score" not in result.columns
    assert "policy_code" not in result.columns
    assert "int_rate" not in result.columns
    assert "grade" not in result.columns
    assert "source" in result.columns
    assert result["source"].nunique() == 2
    assert result["accepted_indicator"].tolist() == [1, 1, 0, 0]


def test_build_shared_features_input_riskscore():
    accepted = pd.DataFrame(
        {
            "loan_amnt": [10000],
            "dti": [15.0],
            "emp_length": [5],
            "addr_state": ["CA"],
            "purpose": ["debt_consolidation"],
            "home_ownership": ["RENT"],
            "annual_inc": [60000],
            "verification_status": ["Verified"],
            "delinq_2yrs": [0],
            "fico_range_low": [680],
            "fico_range_high": [684],
            "open_acc": [10],
            "revol_bal": [15000],
            "revol_util": [45.0],
            "total_acc": [20],
            "issue_d": ["2013-06"],
            "term": [36],
            "zip_code": ["945xx"],
        }
    )
    rejected = pd.DataFrame(
        {
            "Amount Requested": [15000],
            "Debt-To-Income Ratio": [18.5],
            "State": ["CA"],
            "Employment Length": [3],
            "Risk_Score": [680],
            "Application Date": ["2013-07"],
            "Loan Title": ["debt_consolidation"],
            "Zip Code": ["945xx"],
            "Policy Code": [0],
        }
    )

    result = build_shared_features(accepted, rejected, risk_score_setting="input_riskscore")

    assert "risk_score" in result.columns
    assert "policy_code" not in result.columns
    assert result.loc[result["source"] == "rejected", "risk_score"].tolist() == [680]


def test_build_shared_features_anchor_riskscore():
    accepted = pd.DataFrame({"loan_amnt": [10000]})
    rejected = pd.DataFrame({"Amount Requested": [15000], "Risk_Score": [680]})

    result = build_shared_features(accepted, rejected, risk_score_setting="anchor_riskscore")

    assert "risk_score" not in result.columns
    assert "risk_score_anchor" in result.columns
    assert result.loc[result["source"] == "rejected", "risk_score_anchor"].tolist() == [680.0]


def test_build_shared_features_unknown_riskscore_setting_raises():
    accepted = pd.DataFrame({"loan_amnt": [10000]})
    rejected = pd.DataFrame({"Amount Requested": [15000]})

    with pytest.raises(ValueError, match="Unknown risk_score_setting"):
        build_shared_features(accepted, rejected, risk_score_setting="unknown")


def test_build_accepted_rich_features():
    accepted = pd.DataFrame(
        {
            "loan_amnt": [10000],
            "dti": [15.0],
            "emp_length": [5],
            "addr_state": ["CA"],
            "purpose": ["debt_consolidation"],
            "home_ownership": ["RENT"],
            "annual_inc": [60000],
            "verification_status": ["Verified"],
            "delinq_2yrs": [0],
            "fico_range_low": [680],
            "fico_range_high": [684],
            "open_acc": [10],
            "revol_bal": [15000],
            "revol_util": [45.0],
            "total_acc": [20],
            "issue_d": ["2013-06"],
            "term": [36],
            "int_rate": [10.5],
            "installment": [325.0],
            "grade": ["B"],
            "sub_grade": ["B3"],
            "loan_status": ["Fully Paid"],
            "total_pymnt": [10800.0],
        }
    )

    result = build_accepted_rich_features(accepted)

    assert "fico_avg" in result.columns
    assert result["fico_avg"].iloc[0] == 682.0
    assert "grade" in result.columns
    assert "int_rate" in result.columns
    assert "loan_status" not in result.columns
    assert "total_pymnt" not in result.columns
