import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    compute_brier,
    compute_calibration_slope_intercept,
    compute_decile_bad_rate,
    compute_ece,
    compute_ece_equal_width,
    compute_ks,
    compute_psi,
)


def test_brier_perfect():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.0, 1.0, 0.0, 1.0])

    assert compute_brier(y_true, y_pred) < 0.01


def test_brier_worst():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([1.0, 0.0, 1.0, 0.0])

    assert compute_brier(y_true, y_pred) > 0.9


def test_ece_perfect_calibration():
    rng = np.random.default_rng(42)
    y_true = np.array([0, 1, 0, 1, 0, 1] * 10)
    y_pred = np.clip(y_true + rng.normal(0, 0.01, len(y_true)), 0.01, 0.99)

    ece = compute_ece(y_true, y_pred, n_bins=15)

    assert ece < 0.1


def test_ece_overconfident_but_correct():
    y_true = np.array([0, 1] * 30)
    y_pred = np.array([0.0, 1.0] * 30)

    ece = compute_ece(y_true, y_pred, n_bins=15)

    assert ece < 0.01


def test_ece_equal_width_handles_right_edge():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0.0, 1.0, 1.0])

    ece = compute_ece_equal_width(y_true, y_pred, n_bins=10)

    assert ece == 0.0


def test_ks_range():
    y_true = np.array([0] * 50 + [1] * 50)
    y_pred = np.linspace(0, 1, 100)

    ks, threshold = compute_ks(y_true, y_pred)

    assert 0 <= ks <= 1
    assert 0 <= threshold <= 1


def test_calibration_slope():
    y_true = np.array([0, 1, 0, 1] * 25)
    y_pred = y_true.astype(float) * 0.8 + 0.1

    slope, intercept = compute_calibration_slope_intercept(y_true, y_pred)

    assert slope > 0
    assert np.isfinite(intercept)


def test_compute_all_metrics_reports_credit_scoring_fields():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0.05, 0.90, 0.20, 0.80, 0.30, 0.70])

    metrics = compute_all_metrics(y_true, y_pred)

    assert metrics["AUROC"] == 1.0
    assert metrics["PR-AUC_baseline"] == pytest.approx(y_true.mean())
    assert {"KS", "KS_threshold", "Brier", "ECE", "calib_slope", "calib_intercept"}.issubset(metrics)


def test_compute_psi_same_distribution_is_near_zero():
    expected = np.linspace(0.01, 0.99, 100)
    actual = expected.copy()

    assert compute_psi(expected, actual) < 1e-6


def test_compute_decile_bad_rate_returns_decile_report():
    y_true = np.array([0, 1] * 50)
    y_pred = np.linspace(0.01, 0.99, 100)

    report = compute_decile_bad_rate(y_true, y_pred, n_deciles=10)

    assert report["overall_bad_rate"] == 0.5
    assert len(report["deciles"]) == 10
    assert sum(decile["n_samples"] for decile in report["deciles"]) == 100


def test_metric_inputs_must_have_matching_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_brier(np.array([0, 1]), np.array([0.1]))
