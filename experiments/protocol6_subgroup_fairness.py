from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.leakage_audit import audit_features
from src.data.processed_cache import DEFAULT_ACCEPTED_RICH_CACHE_PATH, load_or_build_accepted_labeled_rich
from src.data.splitter import time_split
from src.evaluation.protocol import PROTOCOL6_DEFAULT_GROUP_COLUMNS, run_protocol_6


def main(
    data_path: str,
    output_path: str,
    model_names: list[str] | None = None,
    group_columns: list[str] | None = None,
    target_bad_rate: float = 0.08,
    min_approval_rate: float = 0.20,
    lgd: float = 0.45,
    approval_pd_threshold: float | None = None,
    min_group_size: int = 20,
    test_period: str = "test_normal",
    device_type: str = "cpu",
    gpu_device_id: int = 0,
    cache_path: str | None = str(DEFAULT_ACCEPTED_RICH_CACHE_PATH),
    refresh_cache: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Protocol 6 subgroup robustness/fairness audit and write a metrics CSV."""
    modeling_frame, cache_hit = load_or_build_accepted_labeled_rich(
        data_path,
        cache_path=cache_path,
        refresh_cache=refresh_cache,
    )
    print(f"Protocol 6 data source: {'processed cache' if cache_hit else 'raw CSV'}")

    splits = time_split(modeling_frame, date_col="application_date")
    if test_period not in splits:
        raise KeyError(f"Unknown Protocol 6 test period: {test_period}")

    X_train, y_train, _groups_train, _loan_amounts_train = _make_fairness_matrix(splits["train"])
    X_validation, y_validation, _groups_validation, _loan_amounts_validation = _make_fairness_matrix(
        splits["validation"],
        columns=X_train.columns,
    )
    X_test, y_test, groups_test, loan_amounts_test = _make_fairness_matrix(
        splits[test_period],
        columns=X_train.columns,
    )

    audit_features(X_train)
    audit_features(X_validation)
    audit_features(X_test)

    result_frame = run_protocol_6(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        groups_test,
        loan_amounts_test,
        model_names=model_names,
        group_columns=group_columns,
        target_bad_rate=target_bad_rate,
        min_approval_rate=min_approval_rate,
        lgd=lgd,
        approval_pd_threshold=approval_pd_threshold,
        min_group_size=min_group_size,
        device_type=device_type,
        gpu_device_id=gpu_device_id,
        random_state=random_state,
        verbose=True,
    )
    result_frame.insert(2, "test_period", test_period)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output, index=False)

    _print_summary(result_frame)
    return result_frame


def _make_fairness_matrix(
    frame: pd.DataFrame,
    columns: pd.Index | list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    if len(frame) == 0:
        raise ValueError("Protocol 6 split is empty.")
    if "loan_amnt" not in frame.columns:
        raise KeyError("Protocol 6 requires loan_amnt for subgroup profit metrics.")

    y = frame["default_label"].astype(int).to_numpy()
    loan_amounts = pd.to_numeric(frame["loan_amnt"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    group_features = _build_protocol6_group_features(frame)
    features = frame.drop(columns=["default_label", "application_date", "issue_d"], errors="ignore")
    if columns is None:
        numeric_columns = features.select_dtypes(include=[np.number]).columns
    else:
        numeric_columns = pd.Index(columns)

    X = features.reindex(columns=numeric_columns).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if X.shape[1] == 0:
        raise ValueError("Protocol 6 requires at least one numeric feature.")
    return X.reset_index(drop=True), y, group_features.reset_index(drop=True), loan_amounts


def _build_protocol6_group_features(frame: pd.DataFrame) -> pd.DataFrame:
    groups = pd.DataFrame(index=frame.index)
    state = _first_available(frame, ["state", "addr_state"])
    if state is not None:
        groups["state"] = _clean_string_group(state)

    zip_values = _first_available(frame, ["zip3", "zip_code"])
    if zip_values is not None:
        groups["zip3_region"] = _derive_zip3(zip_values)

    purpose = _first_available(frame, ["loan_purpose", "purpose"])
    if purpose is not None:
        groups["loan_purpose"] = _clean_string_group(purpose)

    emp_length = _first_available(frame, ["employment_length", "emp_length"])
    if emp_length is not None:
        groups["employment_length_group"] = emp_length.map(_employment_length_group).fillna("missing")

    income = _first_available(frame, ["income", "annual_inc"])
    if income is not None:
        groups["income_group"] = _quantile_group(income, prefix="income")

    risk_score = _first_available(frame, ["risk_score", "risk_score_anchor", "Risk_Score"])
    if risk_score is not None:
        groups["risk_score_band"] = _quantile_group(risk_score, prefix="risk")

    return groups[[column for column in PROTOCOL6_DEFAULT_GROUP_COLUMNS if column in groups.columns]]


def _first_available(frame: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    return None


def _clean_string_group(values: pd.Series) -> pd.Series:
    return values.astype(object).where(values.notna(), "missing").astype(str).str.strip().replace("", "missing")


def _derive_zip3(values: pd.Series) -> pd.Series:
    cleaned = _clean_string_group(values)
    return cleaned.map(lambda value: value[:3] if value != "missing" else "missing")


def _employment_length_group(value: object) -> str:
    if pd.isna(value):
        return "missing"
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return "missing"
    if "<" in text:
        return "0"
    match = re.search(r"\d+", text)
    if match is None:
        return "missing"
    years = int(match.group(0))
    if years <= 0:
        return "0"
    if years <= 3:
        return "1-3"
    if years <= 7:
        return "4-7"
    return "8+"


def _quantile_group(values: pd.Series, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series("missing", index=values.index, dtype=object)
    valid = numeric.notna()
    if not valid.any():
        return result
    unique_count = int(numeric[valid].nunique())
    if unique_count < 2:
        result.loc[valid] = f"{prefix}_single"
        return result

    n_bins = min(4, unique_count)
    try:
        binned = pd.qcut(numeric[valid].rank(method="first"), q=n_bins, labels=False)
        result.loc[valid] = [f"{prefix}_Q{int(value) + 1}" for value in binned]
    except ValueError:
        result.loc[valid] = f"{prefix}_single"
    return result


def _parse_model_names(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [name.strip() for name in value.split(",") if name.strip()]


def _parse_group_columns(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [name.strip() for name in value.split(",") if name.strip()]


def _print_summary(result_frame: pd.DataFrame) -> None:
    evaluated = result_frame.loc[result_frame["skip_reason"].isna()]
    for (model, group_feature), group in evaluated.groupby(["model", "group_feature"]):
        first = group.iloc[0]
        print(
            f"{model} | {group_feature}: "
            f"groups={int(first['n_groups_evaluated'])}/{int(first['n_groups_total'])}, "
            f"approval_gap={first['approval_rate_gap']:.4f}, "
            f"bad_gap={first['bad_rate_gap']:.4f}, "
            f"equal_opp_gap={first['equal_opportunity_gap']:.4f}, "
            f"manual_gap={first['manual_review_burden_gap']:.4f}, "
            f"profit_gap={first['profit_gap']:.2f}"
        )

    skipped = result_frame.loc[result_frame["skip_reason"].notna()]
    if len(skipped) > 0:
        counts = skipped.groupby(["model", "group_feature", "skip_reason"]).size()
        for (model, group_feature, reason), count in counts.items():
            print(f"{model} | {group_feature}: skipped {count} subgroup(s) ({reason})")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run Protocol 6 subgroup robustness/fairness audit.")
    parser.add_argument("data_path")
    parser.add_argument("output_path")
    parser.add_argument("--models", default=None, help="Comma-separated model names, e.g. LogisticRegression,LightGBM")
    parser.add_argument(
        "--group-columns",
        default=None,
        help="Comma-separated subgroup columns. Defaults to available Protocol 6 subgroup fields.",
    )
    parser.add_argument("--target-bad-rate", type=float, default=0.08)
    parser.add_argument("--min-approval-rate", type=float, default=0.20)
    parser.add_argument("--lgd", type=float, default=0.45)
    parser.add_argument("--approval-pd-threshold", type=float, default=None)
    parser.add_argument("--min-group-size", type=int, default=20)
    parser.add_argument("--test-period", default="test_normal", choices=["test_normal", "test_extended", "test_structural_break"])
    parser.add_argument("--device-type", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--cache-path", default=str(DEFAULT_ACCEPTED_RICH_CACHE_PATH))
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        output_path=args.output_path,
        model_names=_parse_model_names(args.models),
        group_columns=_parse_group_columns(args.group_columns),
        target_bad_rate=args.target_bad_rate,
        min_approval_rate=args.min_approval_rate,
        lgd=args.lgd,
        approval_pd_threshold=args.approval_pd_threshold,
        min_group_size=args.min_group_size,
        test_period=args.test_period,
        device_type=args.device_type,
        gpu_device_id=args.gpu_device_id,
        cache_path=args.cache_path,
        refresh_cache=args.refresh_cache,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    _cli()
