from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.baselines.traditional import TRADITIONAL_BASELINES
from src.evaluation.metrics import compute_all_metrics, compute_brier, compute_ece, compute_ece_equal_width, compute_psi
from src.models.device import validate_device_type, validate_gpu_device_id


PROTOCOL4_PERIOD_TYPES = {
    "validation": "validation",
    "test_normal": "normal_drift",
    "test_extended": "extended_drift",
    "test_structural_break": "structural_break_stress",
}
PROTOCOL4_MAIN_PERIODS = ("test_normal", "test_extended")


@dataclass
class ProtocolResult:
    protocol: str
    model_name: str
    metrics: dict[str, float]
    predictions: np.ndarray
    true_labels: np.ndarray


def run_protocol_1(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_names: list[str] | None = None,
    random_state: int = 42,
) -> list[ProtocolResult]:
    """Run Protocol 1: accepted-only out-of-time PD benchmark."""
    x_train, y_train = _validate_feature_label_pair("train", X_train, y_train)
    _validate_feature_label_pair("validation", X_val, y_val)
    x_test, y_test = _validate_feature_label_pair("test", X_test, y_test)

    x_test = x_test.reindex(columns=x_train.columns)
    selected_models = model_names or ["LogisticRegression", "LightGBM", "CatBoost"]
    _validate_model_names(selected_models)

    results = []
    for name in selected_models:
        model = TRADITIONAL_BASELINES[name](random_state=random_state)
        model.fit(x_train, y_train)
        predictions = _predict_positive_probability(model, x_test)
        metrics = compute_all_metrics(y_test, predictions)
        results.append(
            ProtocolResult(
                protocol="Protocol1",
                model_name=name,
                metrics=metrics,
                predictions=predictions,
                true_labels=y_test.copy(),
            )
        )
    return results


def run_protocol_4(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    periods: dict[str, tuple[pd.DataFrame, np.ndarray]],
    model_names: list[str] | None = None,
    approval_pd_threshold: float = 0.5,
    device_type: str = "cpu",
    gpu_device_id: int = 0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Protocol 4: policy-shift and temporal-stability evaluation."""
    x_train, y_train = _validate_feature_label_pair("train", X_train, y_train)
    _validate_protocol4_periods(periods)
    if not 0.0 <= float(approval_pd_threshold) <= 1.0:
        raise ValueError("approval_pd_threshold must be in [0, 1].")
    validated_device = validate_device_type(device_type)
    validated_gpu_device_id = validate_gpu_device_id(gpu_device_id)

    selected_models = model_names or ["LogisticRegression", "LightGBM", "CatBoost"]
    _validate_model_names(selected_models)

    rows: list[dict[str, object]] = []
    for name in selected_models:
        model = _build_protocol_model(name, random_state, validated_device, validated_gpu_device_id)
        model.fit(x_train, y_train)
        train_predictions = _predict_positive_probability(model, x_train)

        model_rows: list[dict[str, object]] = []
        for period_name in _ordered_protocol4_periods(periods):
            X_period, y_period = periods[period_name]
            row_base = _protocol4_base_row(
                model_name=name,
                period_name=period_name,
                n_train=len(x_train),
                approval_pd_threshold=approval_pd_threshold,
            )
            if len(pd.DataFrame(X_period)) == 0:
                model_rows.append({**row_base, "n_samples": 0, "skip_reason": "empty_period"})
                continue

            x_period, labels = _validate_feature_label_pair(period_name, X_period, y_period)
            x_period = x_period.reindex(columns=x_train.columns)
            predictions = _predict_positive_probability(model, x_period)
            metrics = _compute_protocol4_metrics(labels, predictions)
            approval = _compute_approval_summary(labels, predictions, approval_pd_threshold)

            model_rows.append(
                {
                    **row_base,
                    "n_samples": int(len(x_period)),
                    "skip_reason": None,
                    "score_mean": float(np.mean(predictions)),
                    "score_std": float(np.std(predictions)),
                    "score_psi_vs_train": compute_psi(train_predictions, predictions),
                    **approval,
                    **metrics,
                }
            )

        rows.extend(_add_protocol4_drift_columns(model_rows))

    return pd.DataFrame(rows)


def _validate_feature_label_pair(name: str, X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    x = pd.DataFrame(X).copy().reset_index(drop=True)
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError(f"{name} labels must be a one-dimensional array.")
    if len(x) != len(labels):
        raise ValueError(f"{name} features and labels must have the same length.")
    if len(x) == 0:
        raise ValueError(f"{name} features and labels must not be empty.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{name} labels must be binary 0/1 values.")
    if not x.columns.is_unique:
        raise ValueError(f"{name} features must have unique column names.")
    return x, labels.astype(int)


def _validate_protocol4_periods(periods: dict[str, tuple[pd.DataFrame, np.ndarray]]) -> None:
    if not periods:
        raise ValueError("Protocol 4 periods must include validation and must not be empty.")
    if "validation" not in periods:
        raise ValueError("Protocol 4 periods must include a non-empty validation period.")
    X_validation, y_validation = periods["validation"]
    if len(pd.DataFrame(X_validation)) == 0 or len(np.asarray(y_validation)) == 0:
        raise ValueError("Protocol 4 validation period must not be empty.")


def _build_protocol_model(
    name: str,
    random_state: int,
    device_type: str,
    gpu_device_id: int,
):
    if name in {"LightGBM", "CatBoost"}:
        return TRADITIONAL_BASELINES[name](
            random_state=random_state,
            device_type=device_type,
            gpu_device_id=gpu_device_id,
        )
    return TRADITIONAL_BASELINES[name](random_state=random_state)


def _ordered_protocol4_periods(periods: dict[str, tuple[pd.DataFrame, np.ndarray]]) -> list[str]:
    ordered = [name for name in PROTOCOL4_PERIOD_TYPES if name in periods]
    extras = sorted(name for name in periods if name not in PROTOCOL4_PERIOD_TYPES)
    return ordered + extras


def _protocol4_base_row(
    model_name: str,
    period_name: str,
    n_train: int,
    approval_pd_threshold: float,
) -> dict[str, object]:
    period_type = PROTOCOL4_PERIOD_TYPES.get(period_name, "custom")
    return {
        "protocol": "Protocol4",
        "model": model_name,
        "period": period_name,
        "period_type": period_type,
        "is_structural_break_stress_test": period_name == "test_structural_break",
        "n_train": int(n_train),
        "approval_pd_threshold": float(approval_pd_threshold),
    }


def _compute_protocol4_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    labels = np.asarray(y_true)
    predictions = np.asarray(y_pred, dtype=float)
    if len(np.unique(labels)) == 2:
        return compute_all_metrics(labels, predictions)

    default_rate = float(labels.mean())
    return {
        "AUROC": float("nan"),
        "PR-AUC": float("nan"),
        "PR-AUC_baseline": default_rate,
        "default_rate": default_rate,
        "KS": float("nan"),
        "KS_threshold": float("nan"),
        "Brier": compute_brier(labels, predictions),
        "ECE": compute_ece(labels, predictions),
        "ECE_equal_width_10": compute_ece_equal_width(labels, predictions, n_bins=10),
        "ECE_equal_width_20": compute_ece_equal_width(labels, predictions, n_bins=20),
        "calib_slope": float("nan"),
        "calib_intercept": float("nan"),
    }


def _compute_approval_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    approval_pd_threshold: float,
) -> dict[str, float]:
    labels = np.asarray(y_true)
    predictions = np.asarray(y_pred, dtype=float)
    approved = predictions <= float(approval_pd_threshold)
    approved_bad_rate = float(labels[approved].mean()) if approved.any() else float("nan")
    return {
        "approval_rate": float(approved.mean()),
        "approved_bad_rate": approved_bad_rate,
    }


def _add_protocol4_drift_columns(model_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    validation_rows = [row for row in model_rows if row["period"] == "validation" and row.get("skip_reason") is None]
    if not validation_rows:
        raise ValueError("Protocol 4 requires a non-empty validation row for drift baselines.")
    validation = validation_rows[0]
    evaluated = [row for row in model_rows if row.get("skip_reason") is None]

    worst_period_auc = _finite_min(row.get("AUROC") for row in evaluated if row["period"] != "validation")
    worst_main_period_auc = _finite_min(
        row.get("AUROC")
        for row in evaluated
        if row["period"] in PROTOCOL4_MAIN_PERIODS
    )

    enriched = []
    for row in model_rows:
        enriched_row = row.copy()
        if row.get("skip_reason") is None:
            enriched_row["Brier_drift_vs_validation"] = float(row["Brier"] - validation["Brier"])
            enriched_row["ECE_drift_vs_validation"] = float(row["ECE"] - validation["ECE"])
            enriched_row["approval_rate_drift_vs_validation"] = float(
                row["approval_rate"] - validation["approval_rate"]
            )
            enriched_row["bad_rate_drift_vs_validation"] = float(row["default_rate"] - validation["default_rate"])
        else:
            enriched_row["Brier_drift_vs_validation"] = float("nan")
            enriched_row["ECE_drift_vs_validation"] = float("nan")
            enriched_row["approval_rate_drift_vs_validation"] = float("nan")
            enriched_row["bad_rate_drift_vs_validation"] = float("nan")
        enriched_row["worst_period_AUROC"] = worst_period_auc
        enriched_row["worst_main_period_AUROC"] = worst_main_period_auc
        enriched.append(enriched_row)
    return enriched


def _finite_min(values) -> float:
    finite_values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(min(finite_values))


def _validate_model_names(model_names: list[str]) -> None:
    if len(model_names) == 0:
        raise ValueError("model_names must not be empty.")
    missing = [name for name in model_names if name not in TRADITIONAL_BASELINES]
    if missing:
        raise KeyError(f"Unknown model name(s): {missing}")


def _predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.ndim == 2:
        if probabilities.shape[1] < 2:
            raise ValueError("predict_proba must return a positive-class probability column.")
        probabilities = probabilities[:, 1]
    if probabilities.ndim != 1:
        raise ValueError("predictions must be a one-dimensional array.")
    if len(probabilities) != len(X):
        raise ValueError("predictions must have the same length as X.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("predictions must contain finite values.")
    return np.clip(probabilities, 0.0, 1.0)
