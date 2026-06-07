from __future__ import annotations

import pandas as pd


RISK_SCORE_SETTINGS = ("no_riskscore", "input_riskscore", "anchor_riskscore")
RISK_SCORE_COLUMN = "risk_score"
RAW_RISK_SCORE_COLUMN = "Risk_Score"
RISK_SCORE_ANCHOR_COLUMN = "risk_score_anchor"
RISK_SCORE_COLUMNS = (RISK_SCORE_COLUMN, RAW_RISK_SCORE_COLUMN, RISK_SCORE_ANCHOR_COLUMN)


class RiskScoreSettingError(ValueError):
    """Raised when a dataframe violates the requested Risk_Score isolation setting."""


def apply_riskscore_setting(df: pd.DataFrame, setting: str = "no_riskscore") -> pd.DataFrame:
    """Apply the project Risk_Score setting.

    - no_riskscore: drop Risk_Score entirely; this is the primary paper setting.
    - input_riskscore: keep it as the standard ``risk_score`` input feature.
    - anchor_riskscore: move it to ``risk_score_anchor`` so it can be used for
      calibration/binning diagnostics, not model input.
    """
    _validate_setting(setting)
    result = pd.DataFrame(df).copy()
    source_column = find_riskscore_column(result)
    if source_column is None:
        result = result.drop(columns=[RISK_SCORE_ANCHOR_COLUMN], errors="ignore")
        return result

    if setting == "no_riskscore":
        return result.drop(columns=list(RISK_SCORE_COLUMNS), errors="ignore")

    if setting == "input_riskscore":
        result[RISK_SCORE_COLUMN] = result[source_column]
        return result.drop(
            columns=[column for column in [RAW_RISK_SCORE_COLUMN, RISK_SCORE_ANCHOR_COLUMN] if column != RISK_SCORE_COLUMN],
            errors="ignore",
        )

    result[RISK_SCORE_ANCHOR_COLUMN] = result[source_column]
    return result.drop(columns=[RISK_SCORE_COLUMN, RAW_RISK_SCORE_COLUMN], errors="ignore")


def audit_riskscore_setting(df: pd.DataFrame, setting: str = "no_riskscore") -> None:
    """Raise if a transformed dataframe violates the requested Risk_Score policy."""
    _validate_setting(setting)
    columns = set(pd.DataFrame(df).columns)
    has_input_score = RISK_SCORE_COLUMN in columns or RAW_RISK_SCORE_COLUMN in columns
    has_anchor_score = RISK_SCORE_ANCHOR_COLUMN in columns

    if setting == "no_riskscore" and (has_input_score or has_anchor_score):
        raise RiskScoreSettingError("No-RiskScore setting must not contain risk_score or risk_score_anchor.")
    if setting == "input_riskscore" and RAW_RISK_SCORE_COLUMN in columns:
        raise RiskScoreSettingError("Input-RiskScore setting must standardize Risk_Score to risk_score.")
    if setting == "input_riskscore" and has_anchor_score:
        raise RiskScoreSettingError("Input-RiskScore setting must not contain risk_score_anchor.")
    if setting == "anchor_riskscore" and has_input_score:
        raise RiskScoreSettingError("Anchor-RiskScore setting must not expose risk_score as model input.")


def find_riskscore_column(df: pd.DataFrame) -> str | None:
    """Return the available Risk_Score column name, preferring standardized risk_score."""
    columns = pd.DataFrame(df).columns
    if RISK_SCORE_COLUMN in columns:
        return RISK_SCORE_COLUMN
    if RAW_RISK_SCORE_COLUMN in columns:
        return RAW_RISK_SCORE_COLUMN
    if RISK_SCORE_ANCHOR_COLUMN in columns:
        return RISK_SCORE_ANCHOR_COLUMN
    return None


def _validate_setting(setting: str) -> None:
    if setting not in RISK_SCORE_SETTINGS:
        raise ValueError(f"Unknown Risk_Score setting: {setting}. Expected one of {RISK_SCORE_SETTINGS}.")
