from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_brier, compute_calibration_slope_intercept, compute_ece


def cross_population_calibration_check(
    teacher,
    X_accepted_val: pd.DataFrame,
    y_accepted_val: np.ndarray,
    X_hidden_rejected: pd.DataFrame,
    y_hidden_rejected: np.ndarray,
) -> dict[str, float | int]:
    """Check whether accepted-set calibration transfers to rejected-like samples.

    Real rejected applicants have no repayment label, so this diagnostic is meant
    for simulated hidden-reject validation/test settings only.
    """
    X_accepted_val = _as_frame("X_accepted_val", X_accepted_val)
    X_hidden_rejected = _as_frame("X_hidden_rejected", X_hidden_rejected)
    y_accepted_val = _as_binary_vector("y_accepted_val", y_accepted_val, required_length=len(X_accepted_val))
    y_hidden_rejected = _as_binary_vector(
        "y_hidden_rejected",
        y_hidden_rejected,
        required_length=len(X_hidden_rejected),
    )
    _require_both_classes("y_accepted_val", y_accepted_val)
    _require_both_classes("y_hidden_rejected", y_hidden_rejected)

    accepted_probs = _predict_default_probability(teacher, X_accepted_val, use_calibrated=True)
    rejected_probs = _predict_default_probability(teacher, X_hidden_rejected, use_calibrated=True)

    accepted_ece = compute_ece(y_accepted_val, accepted_probs)
    rejected_ece = compute_ece(y_hidden_rejected, rejected_probs)
    accepted_brier = compute_brier(y_accepted_val, accepted_probs)
    rejected_brier = compute_brier(y_hidden_rejected, rejected_probs)
    accepted_slope, accepted_intercept = compute_calibration_slope_intercept(y_accepted_val, accepted_probs)
    rejected_slope, rejected_intercept = compute_calibration_slope_intercept(y_hidden_rejected, rejected_probs)

    return {
        "accepted_ece": accepted_ece,
        "rejected_like_ece": rejected_ece,
        "ece_gap": rejected_ece - accepted_ece,
        "accepted_brier": accepted_brier,
        "rejected_like_brier": rejected_brier,
        "brier_gap": rejected_brier - accepted_brier,
        "accepted_calib_slope": accepted_slope,
        "rejected_like_calib_slope": rejected_slope,
        "calib_slope_gap": rejected_slope - accepted_slope,
        "accepted_calib_intercept": accepted_intercept,
        "rejected_like_calib_intercept": rejected_intercept,
        "calib_intercept_gap": rejected_intercept - accepted_intercept,
        "n_accepted": int(len(y_accepted_val)),
        "n_hidden_rejected": int(len(y_hidden_rejected)),
    }


def low_variance_high_error_diagnostic(
    teacher,
    X_hidden_rejected: pd.DataFrame,
    y_hidden_rejected: np.ndarray,
    variance_pct: float = 20.0,
    error_pct: float = 20.0,
) -> dict[str, float | int]:
    """Identify low-variance but high-error hidden rejected samples."""
    variance_pct = _validate_percentile("variance_pct", variance_pct)
    error_pct = _validate_percentile("error_pct", error_pct)
    X_hidden_rejected = _as_frame("X_hidden_rejected", X_hidden_rejected)
    y_hidden_rejected = _as_binary_vector(
        "y_hidden_rejected",
        y_hidden_rejected,
        required_length=len(X_hidden_rejected),
    )

    if not hasattr(teacher, "compute_uncertainty"):
        raise AttributeError("teacher must provide compute_uncertainty.")
    uncertainty = teacher.compute_uncertainty(X_hidden_rejected)
    if "variance" not in uncertainty:
        raise KeyError("teacher uncertainty must include 'variance'.")
    variance = _as_finite_vector("variance", uncertainty["variance"], required_length=len(X_hidden_rejected))

    probs = _predict_default_probability(teacher, X_hidden_rejected, use_calibrated=False)
    residuals = np.abs(probs - y_hidden_rejected)
    variance_threshold = float(np.percentile(variance, variance_pct))
    error_threshold = float(np.percentile(residuals, 100.0 - error_pct))

    confidently_wrong = (variance <= variance_threshold) & (residuals >= error_threshold)
    mean_error = float(residuals[confidently_wrong].mean()) if confidently_wrong.any() else 0.0
    return {
        "confidently_wrong_rate": float(confidently_wrong.mean()),
        "confidently_wrong_mean_error": mean_error,
        "n_confidently_wrong": int(confidently_wrong.sum()),
        "variance_threshold": variance_threshold,
        "error_threshold": error_threshold,
        "mean_rejected_like_error": float(residuals.mean()),
        "n_hidden_rejected": int(len(y_hidden_rejected)),
    }


def _predict_default_probability(teacher, x: pd.DataFrame, use_calibrated: bool) -> np.ndarray:
    if use_calibrated and getattr(teacher, "calibrated", False) and hasattr(teacher, "predict_calibrated"):
        probabilities = teacher.predict_calibrated(x)
    elif hasattr(teacher, "predict_proba"):
        probabilities = teacher.predict_proba(x)
    else:
        raise AttributeError("teacher must provide predict_proba.")

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 2:
        if probabilities.shape[1] < 2:
            raise ValueError("Two-dimensional probability output must contain a positive-class column.")
        probabilities = probabilities[:, 1]
    return np.clip(_as_finite_vector("probabilities", probabilities, required_length=len(x)), 0.0, 1.0)


def _as_frame(name: str, values: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(values).copy()
    if len(frame) == 0:
        raise ValueError(f"{name} must not be empty.")
    return frame


def _as_binary_vector(name: str, values: np.ndarray, required_length: int) -> np.ndarray:
    array = _as_finite_vector(name, values, required_length=required_length)
    if not np.isin(array, [0.0, 1.0]).all():
        raise ValueError(f"{name} must contain binary 0/1 labels.")
    return array.astype(int)


def _as_finite_vector(name: str, values: np.ndarray, required_length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(array) != required_length:
        raise ValueError(f"{name} must have the same length as its feature frame.")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _require_both_classes(name: str, values: np.ndarray) -> None:
    if len(np.unique(values)) != 2:
        raise ValueError(f"{name} must contain both classes for calibration slope.")


def _validate_percentile(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 100.0:
        raise ValueError(f"{name} must be in [0, 100].")
    return value
