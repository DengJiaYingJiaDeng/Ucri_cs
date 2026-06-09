from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.leakage_audit import audit_features
from src.data.processed_cache import DEFAULT_ACCEPTED_RICH_CACHE_PATH, load_or_build_accepted_labeled_rich
from src.data.splitter import time_split
from src.evaluation.protocol import run_protocol_4


def main(
    data_path: str,
    output_path: str,
    model_names: list[str] | None = None,
    approval_pd_threshold: float = 0.5,
    device_type: str = "cpu",
    gpu_device_id: int = 0,
    cache_path: str | None = str(DEFAULT_ACCEPTED_RICH_CACHE_PATH),
    refresh_cache: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Protocol 4 temporal-stability evaluation and write a metrics CSV."""
    modeling_frame, cache_hit = load_or_build_accepted_labeled_rich(
        data_path,
        cache_path=cache_path,
        refresh_cache=refresh_cache,
    )
    print(f"Protocol 4 data source: {'processed cache' if cache_hit else 'raw CSV'}")

    splits = time_split(modeling_frame, date_col="application_date")
    train = splits["train"]
    X_train, y_train = _make_numeric_temporal_matrix(train)
    audit_features(X_train)

    periods: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for period_name in ["validation", "test_normal", "test_extended", "test_structural_break"]:
        X_period, y_period = _make_numeric_temporal_matrix(
            splits[period_name],
            columns=X_train.columns,
            allow_empty=True,
        )
        if len(X_period) > 0:
            audit_features(X_period)
        periods[period_name] = (X_period, y_period)

    result_frame = run_protocol_4(
        X_train,
        y_train,
        periods,
        model_names=model_names,
        approval_pd_threshold=approval_pd_threshold,
        device_type=device_type,
        gpu_device_id=gpu_device_id,
        random_state=random_state,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output, index=False)

    for _, row in result_frame.loc[result_frame["skip_reason"].isna()].iterrows():
        print(
            f"{row['model']} | {row['period']}: "
            f"AUROC={row['AUROC']:.4f}, "
            f"Brier={row['Brier']:.4f}, "
            f"ECE={row['ECE']:.4f}, "
            f"PSI={row['score_psi_vs_train']:.4f}, "
            f"approval_rate={row['approval_rate']:.4f}"
        )

    skipped = result_frame.loc[result_frame["skip_reason"].notna()]
    for _, row in skipped.iterrows():
        print(f"{row['model']} | {row['period']}: skipped ({row['skip_reason']})")

    return result_frame


def _make_numeric_temporal_matrix(
    frame: pd.DataFrame,
    columns: pd.Index | list[str] | None = None,
    allow_empty: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(frame) == 0:
        if allow_empty and columns is not None:
            return pd.DataFrame(columns=pd.Index(columns)), np.array([], dtype=int)
        raise ValueError("Protocol 4 split is empty.")

    y = frame["default_label"].astype(int).to_numpy()
    features = frame.drop(columns=["default_label", "application_date", "issue_d"], errors="ignore")
    if columns is None:
        numeric_columns = features.select_dtypes(include=[np.number]).columns
    else:
        numeric_columns = pd.Index(columns)

    X = features.reindex(columns=numeric_columns).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if X.shape[1] == 0:
        raise ValueError("Protocol 4 requires at least one numeric feature.")
    return X.reset_index(drop=True), y


def _parse_model_names(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [name.strip() for name in value.split(",") if name.strip()]


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run Protocol 4 temporal-stability experiment.")
    parser.add_argument("data_path")
    parser.add_argument("output_path")
    parser.add_argument("--models", default=None, help="Comma-separated model names, e.g. LogisticRegression,LightGBM")
    parser.add_argument("--approval-pd-threshold", type=float, default=0.5)
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
        approval_pd_threshold=args.approval_pd_threshold,
        device_type=args.device_type,
        gpu_device_id=args.gpu_device_id,
        cache_path=args.cache_path,
        refresh_cache=args.refresh_cache,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    _cli()
