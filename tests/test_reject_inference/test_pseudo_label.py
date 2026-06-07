import numpy as np
import pandas as pd
import pytest

from src.reject_inference.pseudo_label import PseudoLabeler


@pytest.fixture
def pseudo_label_data():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(
        {
            "loan_amount": rng.lognormal(9, 0.5, 200),
            "dti": rng.uniform(5, 40, 200),
        }
    )
    teacher_probs = rng.beta(2, 8, 200)
    uncertainty = rng.uniform(0, 1, 200)
    return x, teacher_probs, uncertainty


def test_pseudo_labeler_generates_soft_labels(pseudo_label_data):
    x, probs, uncertainty = pseudo_label_data
    labeler = PseudoLabeler(tau_u=0.5, gamma=2.0)

    result = labeler.label(x, probs, uncertainty)

    assert {"soft_label", "weight", "decision"}.issubset(result)
    assert result["soft_label"].shape == probs.shape
    assert result["weight"].shape == uncertainty.shape
    assert result["decision"].shape == uncertainty.shape


def test_pseudo_labeler_clips_extreme_soft_labels():
    x = pd.DataFrame({"loan_amount": [10000, 20000], "dti": [10, 20]})
    probs = np.array([0.0, 1.0])
    uncertainty = np.array([0.1, 0.1])
    labeler = PseudoLabeler(tau_u=0.5, gamma=2.0)

    result = labeler.label(x, probs, uncertainty)

    assert np.all(result["soft_label"] > 0)
    assert np.all(result["soft_label"] < 1)


def test_pseudo_labeler_weights_decrease_with_uncertainty(pseudo_label_data):
    x, probs, uncertainty = pseudo_label_data
    labeler = PseudoLabeler(tau_u=1.0, gamma=2.0)

    result = labeler.label(x, probs, uncertainty)

    low_unc_mask = uncertainty < 0.3
    high_unc_mask = uncertainty > 0.7
    assert np.mean(result["weight"][low_unc_mask]) > np.mean(result["weight"][high_unc_mask])


def test_pseudo_labeler_threshold_sets_weights_to_zero(pseudo_label_data):
    x, probs, uncertainty = pseudo_label_data
    labeler = PseudoLabeler(tau_u=0.5, gamma=2.0)

    result = labeler.label(x, probs, uncertainty)

    assert np.all(result["weight"][uncertainty >= 0.5] == 0)
    assert np.all(result["weight"][uncertainty < 0.5] > 0)


def test_pseudo_labeler_three_way_decision():
    x = pd.DataFrame({"loan_amount": [10000, 20000, 30000, 40000], "dti": [10, 20, 30, 40]})
    probs = np.array([0.1, 0.5, 0.8, 0.1])
    uncertainty = np.array([0.1, 0.1, 0.1, 0.9])
    labeler = PseudoLabeler(tau_u=0.5, gamma=2.0, theta_low=0.3, theta_high=0.6)

    result = labeler.label(x, probs, uncertainty)

    assert result["decision"].tolist() == ["approve", "manual_review", "reject", "manual_review"]


def test_compute_coverage_uses_positive_weights(pseudo_label_data):
    x, probs, uncertainty = pseudo_label_data
    labeler = PseudoLabeler(tau_u=0.5, gamma=2.0)
    result = labeler.label(x, probs, uncertainty)

    coverage = labeler.compute_coverage(result["weight"])

    assert coverage == pytest.approx((uncertainty < 0.5).mean())


def test_tau_sensitivity_reports_precision_coverage_and_restores_tau(pseudo_label_data):
    x, probs, uncertainty = pseudo_label_data
    true_labels = (probs >= 0.5).astype(int)
    labeler = PseudoLabeler(tau_u=0.4, gamma=2.0)

    report = labeler.tau_sensitivity(x, probs, uncertainty, true_labels, tau_values=[0.2, 0.5])

    assert labeler.tau_u == 0.4
    assert list(report) == ["tau_sensitivity"]
    assert [entry["tau_u"] for entry in report["tau_sensitivity"]] == [0.2, 0.5]
    assert all({"coverage", "precision", "ece"}.issubset(entry) for entry in report["tau_sensitivity"])


def test_coverage_constrained_label_selects_lowest_uncertainty_fraction():
    x = pd.DataFrame({"loan_amount": np.arange(10), "dti": np.arange(10)})
    probs = np.linspace(0.1, 0.9, 10)
    uncertainty = np.linspace(0.0, 0.9, 10)
    labeler = PseudoLabeler(tau_u=0.1, gamma=2.0)

    result = labeler.coverage_constrained_label(x, probs, uncertainty, coverage_target=0.3)

    assert result["coverage"] == pytest.approx(0.3)
    assert np.array_equal(result["weight"] > 0, np.array([True, True, True, False, False, False, False, False, False, False]))


def test_precision_coverage_curve_reports_arrays_and_restores_tau(pseudo_label_data):
    x, probs, uncertainty = pseudo_label_data
    true_labels = (probs >= 0.5).astype(int)
    labeler = PseudoLabeler(tau_u=0.4, gamma=2.0)

    curve = labeler.precision_coverage_curve(x, probs, uncertainty, true_labels, tau_values=[0.1, 0.2, 0.3])

    assert labeler.tau_u == 0.4
    assert set(curve) == {"tau_u", "coverage", "precision", "ece"}
    assert len(curve["coverage"]) == 3
    assert np.all(np.diff(curve["coverage"]) >= 0)


def test_pseudo_labeler_rejects_invalid_input_lengths(pseudo_label_data):
    x, probs, uncertainty = pseudo_label_data
    labeler = PseudoLabeler()

    with pytest.raises(ValueError, match="same length"):
        labeler.label(x, probs[:-1], uncertainty)


def test_pseudo_labeler_rejects_non_normalized_uncertainty(pseudo_label_data):
    x, probs, uncertainty = pseudo_label_data
    uncertainty = uncertainty.copy()
    uncertainty[0] = 1.1
    labeler = PseudoLabeler()

    with pytest.raises(ValueError, match="uncertainty"):
        labeler.label(x, probs, uncertainty)
