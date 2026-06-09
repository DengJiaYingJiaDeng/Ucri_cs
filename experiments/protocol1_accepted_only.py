from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.leakage_audit import audit_features
from src.data.processed_cache import DEFAULT_ACCEPTED_RICH_CACHE_PATH, load_or_build_accepted_labeled_rich
from src.data.splitter import time_split
from src.evaluation.protocol import run_protocol_1


def main(
    data_path: str,
    output_path: str,
    model_names: list[str] | None = None,
    cache_path: str | None = str(DEFAULT_ACCEPTED_RICH_CACHE_PATH),
    refresh_cache: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run accepted-only out-of-time Protocol 1 and write a metrics CSV."""
    modeling_frame, cache_hit = load_or_build_accepted_labeled_rich(
        data_path,
        cache_path=cache_path,
        refresh_cache=refresh_cache,
    )
    print(f"Protocol 1 data source: {'processed cache' if cache_hit else 'raw CSV'}")

    splits = time_split(modeling_frame, date_col="application_date")
    train = splits["train"]
    validation = splits["validation"]
    test = splits["test_normal"]

    X_train, y_train = _make_numeric_protocol_matrix(train)
    X_val, y_val = _make_numeric_protocol_matrix(validation, columns=X_train.columns)
    X_test, y_test = _make_numeric_protocol_matrix(test, columns=X_train.columns)

    audit_features(X_train)
    audit_features(X_val)
    audit_features(X_test)

    results = run_protocol_1(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        model_names=model_names,
        random_state=random_state,
    )

    result_frame = pd.DataFrame([{"model": result.model_name, **result.metrics} for result in results])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output, index=False)

    for result in results:
        print(
            f"{result.model_name}: "
            f"AUROC={result.metrics['AUROC']:.4f}, "
            f"KS={result.metrics['KS']:.4f}, "
            f"Brier={result.metrics['Brier']:.4f}"
        )

    return result_frame


def _make_numeric_protocol_matrix(
    frame: pd.DataFrame,
    columns: pd.Index | list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(frame) == 0:
        raise ValueError("Protocol 1 split is empty.")
    y = frame["default_label"].astype(int).to_numpy()
    features = frame.drop(columns=["default_label", "application_date", "issue_d"], errors="ignore")

    if columns is None:
        numeric_columns = features.select_dtypes(include=[np.number]).columns
    else:
        numeric_columns = pd.Index(columns)

    X = features.reindex(columns=numeric_columns).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if X.shape[1] == 0:
        raise ValueError("Protocol 1 requires at least one numeric feature.")
    return X, y


def _parse_model_names(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [name.strip() for name in value.split(",") if name.strip()]


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run Protocol 1 accepted-only out-of-time PD benchmark.")
    parser.add_argument("data_path")
    parser.add_argument("output_path")
    parser.add_argument("--models", default=None, help="Comma-separated model names, e.g. LogisticRegression,LightGBM")
    parser.add_argument("--cache-path", default=str(DEFAULT_ACCEPTED_RICH_CACHE_PATH))
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        output_path=args.output_path,
        model_names=_parse_model_names(args.models),
        cache_path=args.cache_path,
        refresh_cache=args.refresh_cache,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    _cli()
