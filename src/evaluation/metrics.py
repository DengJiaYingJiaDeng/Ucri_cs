from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def _validate_binary_metric_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    if y_true.shape[0] == 0:
        raise ValueError("Metric inputs must not be empty.")
    return y_true, np.clip(y_pred, 0.0, 1.0)


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 15) -> float:
    """Compute equal-mass Expected Calibration Error."""
    y_true, y_pred = _validate_binary_metric_inputs(y_true, y_pred)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    sorted_indices = np.argsort(y_pred)
    bins = np.array_split(sorted_indices, min(n_bins, len(sorted_indices)))
    ece = 0.0
    for bin_indices in bins:
        if len(bin_indices) == 0:
            continue
        bin_acc = y_true[bin_indices].mean()
        bin_conf = y_pred[bin_indices].mean()
        ece += (len(bin_indices) / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_ece_equal_width(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Compute equal-width Expected Calibration Error."""
    y_true, y_pred = _validate_binary_metric_inputs(y_true, y_pred)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])
        else:
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if not mask.any():
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_pred[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate_binary_metric_inputs(y_true, y_pred)
    return float(brier_score_loss(y_true, y_pred))


def compute_ks(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true, y_pred = _validate_binary_metric_inputs(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ks_values = tpr - fpr
    best_idx = int(np.argmax(ks_values))
    return float(ks_values[best_idx]), float(np.clip(thresholds[best_idx], 0.0, 1.0))


def compute_calibration_slope_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true, y_pred = _validate_binary_metric_inputs(y_true, y_pred)
    y_pred_clipped = np.clip(y_pred, 1e-6, 1 - 1e-6)
    logit_pred = np.log(y_pred_clipped / (1 - y_pred_clipped))

    model = LogisticRegression(C=1e6, fit_intercept=True, solver="lbfgs")
    model.fit(logit_pred.reshape(-1, 1), y_true)
    slope = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    return slope, intercept


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = _validate_binary_metric_inputs(y_true, y_pred)
    try:
        auroc = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        auroc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true, y_pred))
    except ValueError:
        pr_auc = float("nan")

    ks_value, ks_threshold = compute_ks(y_true, y_pred)
    slope, intercept = compute_calibration_slope_intercept(y_true, y_pred)
    default_rate = float(y_true.mean())
    return {
        "AUROC": auroc,
        "PR-AUC": pr_auc,
        "PR-AUC_baseline": default_rate,
        "default_rate": default_rate,
        "KS": ks_value,
        "KS_threshold": ks_threshold,
        "Brier": compute_brier(y_true, y_pred),
        "ECE": compute_ece(y_true, y_pred),
        "ECE_equal_width_10": compute_ece_equal_width(y_true, y_pred, n_bins=10),
        "ECE_equal_width_20": compute_ece_equal_width(y_true, y_pred, n_bins=20),
        "calib_slope": slope,
        "calib_intercept": intercept,
    }


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index using expected-distribution quantile bins."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        raise ValueError("expected and actual must not be empty.")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    bin_edges = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    psi = 0.0
    for i in range(n_bins):
        expected_mask = (expected >= bin_edges[i]) & (expected < bin_edges[i + 1])
        actual_mask = (actual >= bin_edges[i]) & (actual < bin_edges[i + 1])
        expected_fraction = max(float(expected_mask.mean()), 1e-6)
        actual_fraction = max(float(actual_mask.mean()), 1e-6)
        psi += (actual_fraction - expected_fraction) * np.log(actual_fraction / expected_fraction)
    return float(psi)


def compute_decile_bad_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_deciles: int = 10,
) -> dict[str, object]:
    """Report observed bad rate by predicted-score decile."""
    y_true, y_pred = _validate_binary_metric_inputs(y_true, y_pred)
    if n_deciles <= 0:
        raise ValueError("n_deciles must be positive.")

    sorted_indices = np.argsort(y_pred)
    decile_bins = np.array_split(sorted_indices, min(n_deciles, len(sorted_indices)))
    deciles = []
    for i, decile_indices in enumerate(decile_bins, start=1):
        scores = y_pred[decile_indices]
        labels = y_true[decile_indices]
        deciles.append(
            {
                "decile": i,
                "score_range": (float(scores.min()), float(scores.max())),
                "n_samples": int(len(decile_indices)),
                "bad_rate": float(labels.mean()) if len(labels) else 0.0,
            }
        )
    return {"deciles": deciles, "overall_bad_rate": float(y_true.mean())}
