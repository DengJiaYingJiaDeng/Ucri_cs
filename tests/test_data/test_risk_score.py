import pandas as pd
import pytest

from src.data.risk_score import (
    RISK_SCORE_ANCHOR_COLUMN,
    RISK_SCORE_COLUMN,
    RAW_RISK_SCORE_COLUMN,
    RiskScoreSettingError,
    apply_riskscore_setting,
    audit_riskscore_setting,
    find_riskscore_column,
)
from src.data.preprocess import build_shared_features


def test_apply_no_riskscore_drops_score_and_anchor_columns():
    df = pd.DataFrame(
        {
            RISK_SCORE_COLUMN: [680, 620],
            RISK_SCORE_ANCHOR_COLUMN: [680, 620],
            "dti": [10.0, 25.0],
        }
    )

    result = apply_riskscore_setting(df, setting="no_riskscore")

    assert RISK_SCORE_COLUMN not in result.columns
    assert RISK_SCORE_ANCHOR_COLUMN not in result.columns
    assert result["dti"].tolist() == [10.0, 25.0]
    audit_riskscore_setting(result, setting="no_riskscore")


def test_apply_input_riskscore_standardizes_raw_column():
    df = pd.DataFrame({RAW_RISK_SCORE_COLUMN: [700, 640], "dti": [12.0, 32.0]})

    result = apply_riskscore_setting(df, setting="input_riskscore")

    assert RISK_SCORE_COLUMN in result.columns
    assert RAW_RISK_SCORE_COLUMN not in result.columns
    assert result[RISK_SCORE_COLUMN].tolist() == [700, 640]
    audit_riskscore_setting(result, setting="input_riskscore")


def test_apply_anchor_riskscore_keeps_anchor_out_of_model_input():
    df = pd.DataFrame({RISK_SCORE_COLUMN: [710, 590], "loan_amount": [5000, 9000]})

    result = apply_riskscore_setting(df, setting="anchor_riskscore")

    assert RISK_SCORE_COLUMN not in result.columns
    assert RISK_SCORE_ANCHOR_COLUMN in result.columns
    assert result[RISK_SCORE_ANCHOR_COLUMN].tolist() == [710, 590]
    audit_riskscore_setting(result, setting="anchor_riskscore")


def test_find_riskscore_column_prefers_standardized_name():
    df = pd.DataFrame({RISK_SCORE_COLUMN: [1], RAW_RISK_SCORE_COLUMN: [2]})

    assert find_riskscore_column(df) == RISK_SCORE_COLUMN


def test_riskscore_audit_raises_on_policy_violations():
    with pytest.raises(RiskScoreSettingError, match="No-RiskScore"):
        audit_riskscore_setting(pd.DataFrame({RISK_SCORE_COLUMN: [680]}), setting="no_riskscore")

    with pytest.raises(RiskScoreSettingError, match="standardize"):
        audit_riskscore_setting(pd.DataFrame({RAW_RISK_SCORE_COLUMN: [680]}), setting="input_riskscore")

    with pytest.raises(RiskScoreSettingError, match="model input"):
        audit_riskscore_setting(pd.DataFrame({RISK_SCORE_COLUMN: [680]}), setting="anchor_riskscore")


def test_unknown_riskscore_setting_raises():
    with pytest.raises(ValueError, match="Unknown Risk_Score setting"):
        apply_riskscore_setting(pd.DataFrame({RISK_SCORE_COLUMN: [680]}), setting="bad_setting")


def test_build_shared_features_riskscore_settings_match_isolation_policy():
    accepted = pd.DataFrame({"loan_amnt": [10000]})
    rejected = pd.DataFrame({"Amount Requested": [15000], "Risk_Score": [680]})

    no_score = build_shared_features(accepted, rejected, risk_score_setting="no_riskscore")
    input_score = build_shared_features(accepted, rejected, risk_score_setting="input_riskscore")
    anchor_score = build_shared_features(accepted, rejected, risk_score_setting="anchor_riskscore")

    audit_riskscore_setting(no_score, setting="no_riskscore")
    audit_riskscore_setting(input_score, setting="input_riskscore")
    audit_riskscore_setting(anchor_score, setting="anchor_riskscore")
