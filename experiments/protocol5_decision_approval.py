from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.leakage_audit import audit_features
from src.data.processed_cache import DEFAULT_ACCEPTED_RICH_CACHE_PATH, load_or_build_accepted_labeled_rich
from src.data.splitter import time_split
from src.evaluation.protocol import run_protocol_5


def main(
    data_path: str,
    output_path: str,
    model_names: list[str] | None = None,
    target_bad_rates: list[float] | None = None,
    min_approval_rates: list[float] | None = None,
    lgd_values: list[float] | None = None,
    test_period: str = "test_normal",
    device_type: str = "cpu",
    gpu_device_id: int = 0,
    cache_path: str | None = str(DEFAULT_ACCEPTED_RICH_CACHE_PATH),
    refresh_cache: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Protocol 5 decision-aware approval simulation and write a metrics CSV."""
    modeling_frame, cache_hit = load_or_build_accepted_labeled_rich(
        data_path,
        cache_path=cache_path,
        refresh_cache=refresh_cache,
    )
    print(f"Protocol 5 data source: {'processed cache' if cache_hit else 'raw CSV'}")

    splits = time_split(modeling_frame, date_col="application_date")
    if test_period not in splits:
        raise KeyError(f"Unknown Protocol 5 test period: {test_period}")

    X_train, y_train, _ = _make_decision_matrix(splits["train"])
    X_validation, y_validation, _ = _make_decision_matrix(splits["validation"], columns=X_train.columns)
    X_test, y_test, loan_amounts_test = _make_decision_matrix(splits[test_period], columns=X_train.columns)

    audit_features(X_train)
    audit_features(X_validation)
    audit_features(X_test)

    result_frame = run_protocol_5(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        loan_amounts_test,
        model_names=model_names,
        target_bad_rates=target_bad_rates,
        min_approval_rates=min_approval_rates,
        lgd_values=lgd_values,
        device_type=device_type,
        gpu_device_id=gpu_device_id,
        random_state=random_state,
        verbose=True,
    )
    result_frame.insert(2, "test_period", test_period)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output, index=False)

    for _, row in result_frame.iterrows():
        print(
            f"{row['model']} | bad<={row['target_bad_rate']:.2f} | "
            f"approval>={row['min_approval_rate']:.2f} | LGD={row['lgd']:.2f}: "
            f"profit={row['expected_profit']:.2f}, "
            f"approval={row['approval_rate']:.4f}, "
            f"bad={row['realized_bad_rate']:.4f}, "
            f"manual={row['manual_review_rate']:.4f}"
        )

    return result_frame


def _make_decision_matrix(
    frame: pd.DataFrame,
    columns: pd.Index | list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if len(frame) == 0:
        raise ValueError("Protocol 5 split is empty.")
    if "loan_amnt" not in frame.columns:
        raise KeyError("Protocol 5 requires loan_amnt for profit simulation.")

    y = frame["default_label"].astype(int).to_numpy()
    loan_amounts = pd.to_numeric(frame["loan_amnt"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    features = frame.drop(columns=["default_label", "application_date", "issue_d"], errors="ignore")
    if columns is None:
        numeric_columns = features.select_dtypes(include=[np.number]).columns
    else:
        numeric_columns = pd.Index(columns)

    X = features.reindex(columns=numeric_columns).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if X.shape[1] == 0:
        raise ValueError("Protocol 5 requires at least one numeric feature.")
    return X.reset_index(drop=True), y, loan_amounts


def _parse_model_names(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [name.strip() for name in value.split(",") if name.strip()]


def _parse_float_list(value: str | None) -> list[float] | None:
    if value is None or value.strip() == "":
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run Protocol 5 decision-aware approval simulation.")
    parser.add_argument("data_path")
    parser.add_argument("output_path")
    parser.add_argument("--models", default=None, help="Comma-separated model names, e.g. LogisticRegression,LightGBM")
    parser.add_argument("--target-bad-rates", default=None, help="Comma-separated values, e.g. 0.05,0.08,0.10,0.12")
    parser.add_argument("--min-approval-rates", default=None, help="Comma-separated values, e.g. 0.2,0.3,0.4,0.5")
    parser.add_argument("--lgd-values", default=None, help="Comma-separated values, e.g. 0.2,0.35,0.45,0.6,0.75,0.9")
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
        target_bad_rates=_parse_float_list(args.target_bad_rates),
        min_approval_rates=_parse_float_list(args.min_approval_rates),
        lgd_values=_parse_float_list(args.lgd_values),
        test_period=args.test_period,
        device_type=args.device_type,
        gpu_device_id=args.gpu_device_id,
        cache_path=args.cache_path,
        refresh_cache=args.refresh_cache,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    _cli()
