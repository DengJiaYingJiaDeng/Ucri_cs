from __future__ import annotations

import pandas as pd


SPLIT_RANGES = {
    "train": ("2012-01-01", "2014-12-31"),
    "validation": ("2015-01-01", "2015-12-31"),
    "test_normal": ("2016-01-01", "2017-12-31"),
    "test_extended": ("2018-01-01", "2019-12-31"),
    "test_structural_break": ("2020-01-01", "2020-12-31"),
}

SUPPORTED_DATE_FORMATS = ("%Y-%m-%d", "%Y-%m", "%b-%Y")


def time_split(
    df: pd.DataFrame,
    date_col: str = "application_date",
    include_extended: bool = True,
) -> dict[str, pd.DataFrame]:
    """Split records into fixed LendingClub out-of-time periods."""
    if date_col not in df.columns:
        raise KeyError(f"Date column not found: {date_col}")

    parsed_dates = _parse_split_dates(df[date_col], date_col=date_col)
    splits = {}
    for split_name, (start, end) in SPLIT_RANGES.items():
        if split_name == "test_extended" and not include_extended:
            continue
        mask = (parsed_dates >= pd.Timestamp(start)) & (parsed_dates <= pd.Timestamp(end))
        splits[split_name] = df.loc[mask].copy()
    return splits


def split_accepted_rejected(
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    date_col: str = "application_date",
    include_extended: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Split accepted and rejected records using the same fixed time windows."""
    accepted_splits = time_split(accepted, date_col=date_col, include_extended=include_extended)
    rejected_splits = time_split(rejected, date_col=date_col, include_extended=include_extended)
    return {
        split_name: {
            "accepted": accepted_splits[split_name],
            "rejected": rejected_splits[split_name],
        }
        for split_name in accepted_splits
    }


def _parse_split_dates(series: pd.Series, date_col: str) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="raise")

    values = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for date_format in SUPPORTED_DATE_FORMATS:
        missing = parsed.isna()
        if not missing.any():
            break
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], format=date_format, errors="coerce")

    if parsed.isna().any():
        examples = values.loc[parsed.isna()].dropna().head(3).tolist()
        raise ValueError(
            f"Could not parse date column {date_col!r}. "
            f"Supported formats are {SUPPORTED_DATE_FORMATS}; examples: {examples}"
        )
    return parsed
