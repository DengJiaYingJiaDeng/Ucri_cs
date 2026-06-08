from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.protocol3_simulated_rejection import compute_rejection_distribution_comparison
from src.data.leakage_audit import audit_features
from src.data.overlap import overlap_diagnostics
from src.data.preprocess import (
    REJECTED_FEATURE_MAPPING,
    SHARED_FEATURE_MAPPING,
    build_shared_features,
    construct_default_label,
    label_maturity_filter,
)
from src.data.splitter import SPLIT_RANGES, time_split
from src.evaluation.metrics import compute_all_metrics
from src.models.propensity import PropensityModel
from src.models.student import StudentModel
from src.models.teacher import TeacherEnsemble
from src.reject_inference.ssl_trainer import UCRITrainer


DEFAULT_TEACHER_CONFIG = {"n_models": 3, "model_types": ["lightgbm", "catboost", "mlp"]}
DEFAULT_DIAGNOSTIC_SAMPLE_SIZE = 2_000
ACCEPTED_PROTOCOL2_COLUMNS = tuple(dict.fromkeys(["loan_status", *SHARED_FEATURE_MAPPING.keys()]))
REJECTED_PROTOCOL2_COLUMNS = tuple(REJECTED_FEATURE_MAPPING.keys())
ACCEPTED_PROTOCOL2_DTYPES = {
    "loan_amnt": "float64",
    "dti": "float64",
    "annual_inc": "float64",
    "delinq_2yrs": "float64",
    "open_acc": "float64",
    "revol_bal": "float64",
    "revol_util": "float64",
    "total_acc": "float64",
    "fico_range_low": "float64",
    "fico_range_high": "float64",
}
REJECTED_PROTOCOL2_DTYPES = {
    "Amount Requested": "float64",
    "Risk_Score": "float64",
    "Policy Code": "float64",
}


def run_protocol_2(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    X_real_rejected: pd.DataFrame,
    teacher_config: dict | None = None,
    student_model_type: str = "lightgbm",
    tau_u: float = 0.5,
    gamma: float = 2.0,
    lambda_distill: float = 0.3,
    random_state: int = 42,
    device_type: str = "cpu",
    gpu_device_id: int = 0,
    diagnostic_sample_size: int | None = DEFAULT_DIAGNOSTIC_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Run Protocol 2 with real rejected applicants as unlabeled samples.

    Real rejected applicants have no repayment labels, so metrics are computed
    only on future accepted test rows. Rejected rows receive diagnostics only.
    """
    x_train, y_train, x_test, y_test, x_rejected = _validate_protocol_inputs(
        X_train,
        y_train,
        X_test,
        y_test,
        X_real_rejected,
    )
    audit_features(x_train)
    audit_features(x_test)
    audit_features(x_rejected)

    _validate_optional_row_limit("diagnostic_sample_size", diagnostic_sample_size)
    diagnostics = _real_rejected_distribution_diagnostics(
        x_train,
        x_rejected,
        random_state=random_state,
        diagnostic_sample_size=diagnostic_sample_size,
    )
    teacher_settings = dict(teacher_config or DEFAULT_TEACHER_CONFIG)
    teacher_settings.setdefault("device_type", device_type)
    teacher_settings.setdefault("gpu_device_id", gpu_device_id)

    trainer = UCRITrainer(
        teacher_config=teacher_settings,
        student_model_type=student_model_type,
        tau_u=tau_u,
        gamma=gamma,
        lambda_distill=lambda_distill,
        random_state=random_state,
        device_type=device_type,
        gpu_device_id=gpu_device_id,
    )
    ucri_result = trainer.run(x_train, y_train, x_rejected)

    accepted_only = StudentModel(
        model_type=student_model_type,
        random_state=random_state,
        device_type=device_type,
        gpu_device_id=gpu_device_id,
    )
    accepted_only.fit(x_train, y_train)

    rows = []
    base_row = {
        "protocol": "Protocol2",
        "evaluation_population": "future_accepted_test",
        "real_rejected_label_available": False,
        "n_train_accepted": int(len(x_train)),
        "n_test_accepted": int(len(x_test)),
        "n_real_rejected_unlabeled": int(len(x_rejected)),
        "accepted_train_bad_rate": float(y_train.mean()),
        "accepted_test_bad_rate": float(y_test.mean()),
        "tau_u": float(tau_u),
        "gamma": float(gamma),
        "lambda_distill": float(lambda_distill),
        "device_type": device_type,
        "gpu_device_id": int(gpu_device_id),
        **diagnostics,
    }

    ucri_test_predictions = ucri_result["student"].predict_proba(x_test)
    ucri_rejected_predictions = ucri_result["student"].predict_proba(x_rejected)
    rows.append(
        _metric_row(
            base_row,
            "UCRI-CS",
            y_test,
            ucri_test_predictions,
            real_rejected_predictions=ucri_rejected_predictions,
            uncertainty=ucri_result["uncertainty"],
            pseudo_labels=ucri_result["pseudo_labels"],
        )
    )

    teacher_test_predictions = _predict_teacher(ucri_result["teacher"], x_test)
    rows.append(
        _metric_row(
            base_row,
            "teacher",
            y_test,
            teacher_test_predictions,
            real_rejected_predictions=ucri_result["teacher_probs"],
            uncertainty=ucri_result["uncertainty"],
        )
    )

    accepted_only_test_predictions = accepted_only.predict_proba(x_test)
    accepted_only_rejected_predictions = accepted_only.predict_proba(x_rejected)
    rows.append(
        _metric_row(
            base_row,
            "accepted-only",
            y_test,
            accepted_only_test_predictions,
            real_rejected_predictions=accepted_only_rejected_predictions,
        )
    )

    return pd.DataFrame(rows)


def _load_protocol2_accepted(path: str, max_rows_per_split: int | None, chunksize: int = 500_000) -> pd.DataFrame:
    frames_by_split = {"train": [], "test_normal": []}
    counts = {split_name: 0 for split_name in frames_by_split}
    for chunk in pd.read_csv(
        path,
        usecols=lambda column: column in ACCEPTED_PROTOCOL2_COLUMNS,
        dtype=ACCEPTED_PROTOCOL2_DTYPES,
        chunksize=chunksize,
    ):
        labeled = construct_default_label(label_maturity_filter(chunk)).dropna(subset=["default_label"])
        for split_name in frames_by_split:
            remaining = None if max_rows_per_split is None else max_rows_per_split - counts[split_name]
            if remaining is not None and remaining <= 0:
                continue
            selected = _take_split_rows(labeled, "issue_d", split_name, remaining)
            if not selected.empty:
                frames_by_split[split_name].append(selected)
                counts[split_name] += len(selected)
        if max_rows_per_split is not None and all(count >= max_rows_per_split for count in counts.values()):
            break

    frames = [frame for split_frames in frames_by_split.values() for frame in split_frames]
    if not frames:
        return pd.DataFrame(columns=ACCEPTED_PROTOCOL2_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _load_protocol2_rejected(
    path: str,
    max_rows: int | None,
    rejected_split: str,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    frames = []
    count = 0
    for chunk in pd.read_csv(
        path,
        usecols=lambda column: column in REJECTED_PROTOCOL2_COLUMNS,
        dtype=REJECTED_PROTOCOL2_DTYPES,
        chunksize=chunksize,
    ):
        remaining = None if max_rows is None else max_rows - count
        if remaining is not None and remaining <= 0:
            break
        selected = _take_split_rows(chunk, "Application Date", rejected_split, remaining)
        if not selected.empty:
            frames.append(selected)
            count += len(selected)

    if not frames:
        return pd.DataFrame(columns=REJECTED_PROTOCOL2_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _take_split_rows(frame: pd.DataFrame, date_col: str, split_name: str, max_rows: int | None) -> pd.DataFrame:
    start, end = SPLIT_RANGES[split_name]
    parsed_dates = _parse_lendingclub_dates(frame[date_col])
    mask = (parsed_dates >= pd.Timestamp(start)) & (parsed_dates <= pd.Timestamp(end))
    selected = frame.loc[mask]
    if max_rows is not None:
        selected = selected.head(max_rows)
    return selected.copy()


def _parse_lendingclub_dates(values: pd.Series) -> pd.Series:
    values = pd.Series(values)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    for date_format in ("%Y-%m-%d", "%Y-%m", "%b-%Y"):
        missing = parsed.isna()
        if not missing.any():
            break
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], format=date_format, errors="coerce")

    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], errors="coerce")
    return parsed


def main(
    accepted_path: str,
    rejected_path: str,
    output_path: str,
    teacher_config: dict | None = None,
    student_model_type: str = "lightgbm",
    risk_score_setting: str = "no_riskscore",
    rejected_split: str = "train",
    tau_u: float = 0.5,
    gamma: float = 2.0,
    lambda_distill: float = 0.3,
    max_accepted_rows: int | None = 10_000,
    max_rejected_rows: int | None = 10_000,
    random_state: int = 42,
    device_type: str = "cpu",
    gpu_device_id: int = 0,
    diagnostic_sample_size: int | None = DEFAULT_DIAGNOSTIC_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Run Protocol 2 on LendingClub accepted/rejected raw CSVs."""
    _validate_optional_row_limit("max_accepted_rows", max_accepted_rows)
    _validate_optional_row_limit("max_rejected_rows", max_rejected_rows)
    if rejected_split not in SPLIT_RANGES:
        raise ValueError(f"rejected_split must be one of {sorted(SPLIT_RANGES)}.")

    accepted = _load_protocol2_accepted(accepted_path, max_accepted_rows)
    rejected = _load_protocol2_rejected(rejected_path, max_rejected_rows, rejected_split)
    labeled = construct_default_label(label_maturity_filter(accepted)).dropna(subset=["default_label"]).copy()

    combined = build_shared_features(labeled, rejected, risk_score_setting=risk_score_setting)

    accepted_shared = combined[combined["source"].eq("accepted")].copy().reset_index(drop=True)
    rejected_shared = combined[combined["source"].eq("rejected")].copy().reset_index(drop=True)
    accepted_shared["default_label"] = labeled["default_label"].astype(int).to_numpy()

    accepted_splits = time_split(accepted_shared, date_col="application_date")
    rejected_splits = time_split(rejected_shared, date_col="application_date")
    if rejected_split not in rejected_splits:
        raise ValueError(f"rejected_split must be one of {sorted(rejected_splits)}.")

    train = accepted_splits["train"].copy()
    test = accepted_splits["test_normal"].copy()
    real_rejected = rejected_splits[rejected_split].copy()

    X_train, y_train = _make_feature_label_matrix(train)
    X_test, y_test = _make_feature_label_matrix(test, columns=X_train.columns)
    X_real_rejected = _make_unlabeled_feature_matrix(real_rejected, columns=X_train.columns)

    result_frame = run_protocol_2(
        X_train,
        y_train,
        X_test,
        y_test,
        X_real_rejected,
        teacher_config=teacher_config,
        student_model_type=student_model_type,
        tau_u=tau_u,
        gamma=gamma,
        lambda_distill=lambda_distill,
        random_state=random_state,
        device_type=device_type,
        gpu_device_id=gpu_device_id,
        diagnostic_sample_size=diagnostic_sample_size,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output, index=False)

    for _, row in result_frame.iterrows():
        print(
            f"{row['model']}: accepted-test AUROC={row['AUROC']:.4f}, "
            f"Brier={row['Brier']:.4f}, ECE={row['ECE']:.4f}, "
            f"real-rejected PD mean={row['real_rejected_mean_pd']:.4f}, "
            f"pseudo coverage={row['pseudo_label_coverage']:.4f}"
        )

    return result_frame


def _real_rejected_distribution_diagnostics(
    X_train: pd.DataFrame,
    X_real_rejected: pd.DataFrame,
    random_state: int,
    diagnostic_sample_size: int | None = DEFAULT_DIAGNOSTIC_SAMPLE_SIZE,
) -> dict[str, float | int]:
    train_sample = _sample_diagnostic_frame(X_train, diagnostic_sample_size, random_state)
    rejected_sample = _sample_diagnostic_frame(X_real_rejected, diagnostic_sample_size, random_state + 1)

    propensity_model = PropensityModel(model_type="logistic", random_state=random_state)
    x_all = pd.concat([train_sample, rejected_sample], ignore_index=True)
    accepted_indicator = np.concatenate(
        [
            np.ones(len(train_sample), dtype=int),
            np.zeros(len(rejected_sample), dtype=int),
        ]
    )
    propensity_model.fit(x_all, accepted_indicator)
    rejected_propensity = propensity_model.predict_proba(rejected_sample)
    overlap = overlap_diagnostics(
        train_sample,
        rejected_sample,
        rejected_propensity,
        k=min(10, len(train_sample)),
    )
    distribution = compute_rejection_distribution_comparison(
        train_sample,
        rejected_sample,
        random_state=random_state,
    )
    return {
        "real_rejected_diagnostic_train_sample_size": int(len(train_sample)),
        "real_rejected_diagnostic_rejected_sample_size": int(len(rejected_sample)),
        "real_rejected_mean_acceptance_propensity": float(rejected_propensity.mean()),
        "real_rejected_median_acceptance_propensity": float(np.median(rejected_propensity)),
        "real_rejected_overlap_coverage": float(overlap["coverage"]),
        "real_rejected_out_of_support_rate": float(overlap["out_of_support_rate"]),
        "real_rejected_n_in_overlap": int(overlap["n_in_overlap"]),
        "real_rejected_n_out_of_support": int(overlap["n_out_of_support"]),
        "real_rejected_mmd_rbf": float(distribution["mmd_rbf"]),
        "real_rejected_mean_pairwise_distance": float(distribution["mean_pairwise_distance"]),
    }


def _sample_diagnostic_frame(frame: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    frame = pd.DataFrame(frame)
    if max_rows is None or len(frame) <= max_rows:
        return frame.copy().reset_index(drop=True)
    return frame.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def _metric_row(
    base_row: dict[str, object],
    model_name: str,
    y_test: np.ndarray,
    test_predictions: np.ndarray,
    real_rejected_predictions: np.ndarray,
    uncertainty: np.ndarray | None = None,
    pseudo_labels: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    rejected_predictions = np.clip(np.asarray(real_rejected_predictions, dtype=float), 0.0, 1.0)
    diagnostics = _real_rejected_prediction_diagnostics(rejected_predictions)
    diagnostics.update(_uncertainty_diagnostics(uncertainty))
    diagnostics.update(_pseudo_label_diagnostics(pseudo_labels))
    return {
        **base_row,
        "model": model_name,
        **compute_all_metrics(y_test, np.clip(np.asarray(test_predictions, dtype=float), 0.0, 1.0)),
        **diagnostics,
    }


def _real_rejected_prediction_diagnostics(probabilities: np.ndarray) -> dict[str, float]:
    return {
        "real_rejected_mean_pd": float(probabilities.mean()),
        "real_rejected_median_pd": float(np.median(probabilities)),
        "real_rejected_p90_pd": float(np.quantile(probabilities, 0.90)),
        "real_rejected_high_pd_rate": float((probabilities >= 0.5).mean()),
    }


def _uncertainty_diagnostics(uncertainty: np.ndarray | None) -> dict[str, float]:
    if uncertainty is None:
        return {
            "real_rejected_mean_uncertainty": float("nan"),
            "real_rejected_median_uncertainty": float("nan"),
            "real_rejected_high_uncertainty_rate": float("nan"),
        }
    values = np.asarray(uncertainty, dtype=float)
    median = float(np.median(values))
    return {
        "real_rejected_mean_uncertainty": float(values.mean()),
        "real_rejected_median_uncertainty": median,
        "real_rejected_high_uncertainty_rate": float((values > median).mean()),
    }


def _pseudo_label_diagnostics(pseudo_labels: dict[str, np.ndarray] | None) -> dict[str, float]:
    if pseudo_labels is None:
        return {
            "pseudo_label_coverage": float("nan"),
            "pseudo_label_mean_weight": float("nan"),
            "pseudo_label_mean_soft_label": float("nan"),
            "pseudo_label_high_risk_rate": float("nan"),
        }
    weights = np.asarray(pseudo_labels["weight"], dtype=float)
    soft_labels = np.asarray(pseudo_labels["soft_label"], dtype=float)
    selected = weights > 0
    return {
        "pseudo_label_coverage": float(selected.mean()),
        "pseudo_label_mean_weight": float(weights[selected].mean()) if selected.any() else 0.0,
        "pseudo_label_mean_soft_label": float(soft_labels[selected].mean()) if selected.any() else float("nan"),
        "pseudo_label_high_risk_rate": float((soft_labels[selected] >= 0.5).mean()) if selected.any() else float("nan"),
    }


def _predict_teacher(teacher: TeacherEnsemble, X: pd.DataFrame) -> np.ndarray:
    probabilities = teacher.predict_calibrated(X) if teacher.calibrated else teacher.predict_proba(X)
    return np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)


def _validate_protocol_inputs(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    X_real_rejected: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    x_train = pd.DataFrame(X_train).copy().reset_index(drop=True)
    x_test = pd.DataFrame(X_test).copy().reset_index(drop=True).reindex(columns=x_train.columns)
    x_rejected = pd.DataFrame(X_real_rejected).copy().reset_index(drop=True).reindex(columns=x_train.columns)
    y_train = _validate_binary_labels("y_train", y_train, expected_length=len(x_train))
    y_test = _validate_binary_labels("y_test", y_test, expected_length=len(x_test))

    if len(x_train) == 0:
        raise ValueError("X_train must not be empty.")
    if len(x_test) == 0:
        raise ValueError("X_test must not be empty.")
    if len(x_rejected) == 0:
        raise ValueError("X_real_rejected must not be empty.")
    if not x_train.columns.is_unique:
        raise ValueError("X_train feature names must be unique.")
    _require_both_classes("y_train", y_train)
    _require_both_classes("y_test", y_test)
    return x_train, y_train, x_test, y_test, x_rejected


def _make_feature_label_matrix(
    frame: pd.DataFrame,
    columns: pd.Index | list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(frame) == 0:
        raise ValueError("Accepted split is empty.")
    y = frame["default_label"].astype(int).to_numpy()
    X = _make_unlabeled_feature_matrix(frame, columns=columns)
    return X, y


def _make_unlabeled_feature_matrix(
    frame: pd.DataFrame,
    columns: pd.Index | list[str] | None = None,
) -> pd.DataFrame:
    features = pd.DataFrame(frame).drop(
        columns=["default_label", "source", "accepted_indicator", "application_date"],
        errors="ignore",
    )
    if columns is not None:
        features = features.reindex(columns=pd.Index(columns))
    if features.shape[1] == 0:
        raise ValueError("Protocol 2 requires at least one shared feature.")
    return features.reset_index(drop=True)


def _validate_binary_labels(name: str, values: np.ndarray, expected_length: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if len(labels) != expected_length:
        raise ValueError(f"{name} must have the same length as its feature frame.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{name} must contain binary 0/1 labels.")
    return labels.astype(int)


def _require_both_classes(name: str, values: np.ndarray) -> None:
    if len(np.unique(values)) != 2:
        raise ValueError(f"{name} must contain both classes.")


def _validate_optional_row_limit(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"{name} must be positive or None.")


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value.strip().lower() == "none":
        return None
    return int(value)


def _parse_teacher_config(value: str | None) -> dict | None:
    if value is None or not value.strip():
        return None
    model_types = [model_type.strip() for model_type in value.split(",") if model_type.strip()]
    if not model_types:
        raise ValueError("teacher model types must not be empty.")
    return {"n_models": len(model_types), "model_types": model_types}


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run Protocol 2 real rejected semi-supervised experiment.")
    parser.add_argument("accepted_path")
    parser.add_argument("rejected_path")
    parser.add_argument("output_path")
    parser.add_argument("--teacher-model-types", default=None)
    parser.add_argument("--student-model-type", default="lightgbm")
    parser.add_argument("--risk-score-setting", default="no_riskscore")
    parser.add_argument("--rejected-split", default="train")
    parser.add_argument("--tau-u", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--lambda-distill", type=float, default=0.3)
    parser.add_argument("--max-accepted-rows", default="10000")
    parser.add_argument("--max-rejected-rows", default="10000")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device-type", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--diagnostic-sample-size", default=str(DEFAULT_DIAGNOSTIC_SAMPLE_SIZE))
    args = parser.parse_args()

    main(
        accepted_path=args.accepted_path,
        rejected_path=args.rejected_path,
        output_path=args.output_path,
        teacher_config=_parse_teacher_config(args.teacher_model_types),
        student_model_type=args.student_model_type,
        risk_score_setting=args.risk_score_setting,
        rejected_split=args.rejected_split,
        tau_u=args.tau_u,
        gamma=args.gamma,
        lambda_distill=args.lambda_distill,
        max_accepted_rows=_parse_optional_int(args.max_accepted_rows),
        max_rejected_rows=_parse_optional_int(args.max_rejected_rows),
        random_state=args.random_state,
        device_type=args.device_type,
        gpu_device_id=args.gpu_device_id,
        diagnostic_sample_size=_parse_optional_int(args.diagnostic_sample_size),
    )


if __name__ == "__main__":
    _cli()
