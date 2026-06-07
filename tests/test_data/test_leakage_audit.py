import pandas as pd
import pytest

from src.data.leakage_audit import ForbiddenFeatureError, audit_features


def test_clean_dataframe_passes_audit():
    df = pd.DataFrame({"loan_amnt": [10000], "dti": [15.0], "emp_length": [5]})

    audit_features(df)


def test_dataframe_with_forbidden_raises():
    df = pd.DataFrame({"loan_amnt": [10000], "total_pymnt": [5000], "dti": [15.0]})

    with pytest.raises(ForbiddenFeatureError, match="total_pymnt"):
        audit_features(df)


def test_dataframe_with_recoveries_raises():
    df = pd.DataFrame({"recoveries": [0], "dti": [15.0]})

    with pytest.raises(ForbiddenFeatureError, match="recoveries"):
        audit_features(df)


def test_dataframe_with_label_proxy_raises():
    df = pd.DataFrame({"loan_status": ["Fully Paid"], "loan_amnt": [10000]})

    with pytest.raises(ForbiddenFeatureError, match="loan_status"):
        audit_features(df)
