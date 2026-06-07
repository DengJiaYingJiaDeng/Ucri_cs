import numpy as np
import pandas as pd
import pytest

from src.baselines.reject_inference import (
    REJECT_INFERENCE_BASELINES,
    domain_adversarial_balancing,
    extrapolation_reject_inference,
    fuzzy_augmentation,
    hard_augmentation,
    mean_teacher_baseline,
    noisy_student_baseline,
    ipw_weighted_pd,
    parceling,
    self_training,
    ssvm_reject_inference,
)


class RecordingModel:
    def __init__(self, probability_batches=None):
        self.probability_batches = list(probability_batches or [])
        self.fit_calls = []

    def fit(self, X, y, sample_weight=None):
        self.fit_calls.append(
            {
                "X": pd.DataFrame(X).reset_index(drop=True),
                "y": np.asarray(y).copy(),
                "sample_weight": None if sample_weight is None else np.asarray(sample_weight, dtype=float).copy(),
            }
        )
        return self

    def predict_proba(self, X):
        n_rows = len(pd.DataFrame(X))
        if self.probability_batches:
            probabilities = np.asarray(self.probability_batches.pop(0), dtype=float)
        else:
            probabilities = np.full(n_rows, 0.5, dtype=float)
        if len(probabilities) != n_rows:
            raise AssertionError("probability batch length does not match X")
        return np.column_stack([1.0 - probabilities, probabilities])


class RecordingPropensity:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict_proba(self, X):
        assert len(pd.DataFrame(X)) == len(np.asarray(self.probabilities))
        return self.probabilities


@pytest.fixture
def reject_data():
    x_labeled = pd.DataFrame(
        {
            "loan_amount": [10000, 12000, 8000, 15000],
            "dti": [10, 15, 20, 25],
            "fico_avg": [720, 690, 660, 640],
        }
    )
    y_labeled = np.array([0, 0, 1, 1])
    x_unlabeled = pd.DataFrame(
        {
            "loan_amount": [9000, 18000, 7000],
            "dti": [12, 28, 18],
            "fico_avg": [700, 630, 680],
        }
    )
    return x_labeled, y_labeled, x_unlabeled


def test_reject_inference_registry_contains_task17_and_task26_baselines():
    expected = {
        "hard",
        "fuzzy",
        "parceling",
        "self_training",
        "ipw",
        "extrapolation",
        "domain_adversarial",
        "ssvm",
        "mean_teacher",
        "noisy_student",
    }

    assert expected.issubset(REJECT_INFERENCE_BASELINES)
    assert all(callable(builder) for builder in REJECT_INFERENCE_BASELINES.values())


def test_hard_augmentation_generates_thresholded_rejected_labels(reject_data):
    x_labeled, y_labeled, x_unlabeled = reject_data
    model = RecordingModel(probability_batches=[np.array([0.2, 0.7, 0.5])])

    result = hard_augmentation(model, x_labeled, y_labeled, x_unlabeled, threshold=0.5)

    assert result is model
    assert len(model.fit_calls) == 2
    augmented = model.fit_calls[-1]
    assert len(augmented["X"]) == len(x_labeled) + len(x_unlabeled)
    assert augmented["y"].tolist() == [0, 0, 1, 1, 0, 1, 1]
    assert augmented["sample_weight"] is None


def test_fuzzy_augmentation_uses_prediction_weighted_soft_parcels(reject_data):
    x_labeled, y_labeled, x_unlabeled = reject_data
    probabilities = np.array([0.2, 0.7, 0.5])
    model = RecordingModel(probability_batches=[probabilities])

    fuzzy_augmentation(model, x_labeled, y_labeled, x_unlabeled)

    augmented = model.fit_calls[-1]
    assert len(augmented["X"]) == len(x_labeled) + 2 * len(x_unlabeled)
    assert augmented["y"].tolist() == [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]
    assert np.allclose(augmented["sample_weight"][: len(x_labeled)], 1.0)
    assert np.allclose(augmented["sample_weight"][len(x_labeled) : len(x_labeled) + len(x_unlabeled)], 1 - probabilities)
    assert np.allclose(augmented["sample_weight"][-len(x_unlabeled) :], probabilities)


def test_parceling_assigns_bin_level_default_rates(reject_data):
    x_labeled, y_labeled, x_unlabeled = reject_data
    probabilities = np.array([0.1, 0.8, 0.9])
    model = RecordingModel(probability_batches=[probabilities])

    parceling(model, x_labeled, y_labeled, x_unlabeled, n_bins=2)

    augmented = model.fit_calls[-1]
    assert len(augmented["X"]) == len(x_labeled) + 2 * len(x_unlabeled)
    unlabeled_positive_weights = augmented["sample_weight"][-len(x_unlabeled) :]
    assert np.all((unlabeled_positive_weights >= 0) & (unlabeled_positive_weights <= 1))
    assert unlabeled_positive_weights[0] < unlabeled_positive_weights[-1]


def test_self_training_adds_only_confident_unlabeled_rows(reject_data):
    x_labeled, y_labeled, x_unlabeled = reject_data
    model = RecordingModel(probability_batches=[np.array([0.1, 0.55, 0.9]), np.array([0.5])])

    self_training(model, x_labeled, y_labeled, x_unlabeled, n_iterations=2, confidence_threshold=0.8)

    augmented = model.fit_calls[-1]
    assert len(augmented["X"]) == len(x_labeled) + 2
    assert augmented["y"].tolist() == [0, 0, 1, 1, 0, 1]


def test_self_training_stops_when_no_confident_rows(reject_data):
    x_labeled, y_labeled, x_unlabeled = reject_data
    model = RecordingModel(probability_batches=[np.array([0.45, 0.55, 0.6])])

    self_training(model, x_labeled, y_labeled, x_unlabeled, n_iterations=3, confidence_threshold=0.8)

    assert len(model.fit_calls) == 1
    assert len(model.fit_calls[-1]["X"]) == len(x_labeled)


def test_ipw_weighted_pd_uses_inverse_propensity_weights(reject_data):
    x_labeled, y_labeled, _ = reject_data
    propensity = RecordingPropensity(np.array([0.5, 0.25, 0.1, 0.01]))
    pd_model = RecordingModel()

    ipw_weighted_pd(propensity, pd_model, x_labeled, y_labeled, eps=0.05)

    weights = pd_model.fit_calls[-1]["sample_weight"]
    assert np.allclose(weights, [2.0, 4.0, 10.0, 20.0])


def test_ipw_weighted_pd_accepts_sklearn_two_column_propensity(reject_data):
    x_labeled, y_labeled, _ = reject_data
    accepted_probabilities = np.array([0.8, 0.5, 0.25, 0.1])
    propensity = RecordingPropensity(np.column_stack([1 - accepted_probabilities, accepted_probabilities]))
    pd_model = RecordingModel()

    ipw_weighted_pd(propensity, pd_model, x_labeled, y_labeled, eps=0.05)

    assert np.allclose(pd_model.fit_calls[-1]["sample_weight"], 1.0 / accepted_probabilities)


def test_extrapolation_adds_highest_risk_rejected_as_bad_labels(reject_data):
    x_labeled, y_labeled, x_unlabeled = reject_data
    model = RecordingModel(probability_batches=[np.array([0.1, 0.95, 0.7])])

    result = extrapolation_reject_inference(model, x_labeled, y_labeled, x_unlabeled, quantile=1 / 3)

    assert result is model
    assert len(model.fit_calls) == 2
    augmented = model.fit_calls[-1]
    assert len(augmented["X"]) == len(x_labeled) + 1
    assert augmented["y"].tolist() == [0, 0, 1, 1, 1]
    assert augmented["X"].iloc[-1]["loan_amount"] == 18000


def test_domain_adversarial_balancing_fits_pd_model_with_domain_weights(reject_data):
    x_labeled, y_labeled, x_unlabeled = reject_data
    x_labeled = x_labeled.copy()
    x_unlabeled = x_unlabeled.copy()
    x_labeled["state"] = ["CA", "CA", "NY", "NY"]
    x_unlabeled["state"] = ["TX", "TX", "CA"]
    pd_model = RecordingModel()

    result = domain_adversarial_balancing(pd_model, x_labeled, y_labeled, x_unlabeled)

    assert result is pd_model
    weights = pd_model.fit_calls[-1]["sample_weight"]
    assert weights.shape == (len(x_labeled),)
    assert np.all(np.isfinite(weights))
    assert np.all((weights >= 0.1) & (weights <= 10.0))


def test_ssvm_reject_inference_returns_predict_proba_model(reject_data):
    x_labeled, y_labeled, x_unlabeled = reject_data
    x_labeled = x_labeled.copy()
    x_unlabeled = x_unlabeled.copy()
    x_labeled["state"] = ["CA", "CA", "NY", "NY"]
    x_unlabeled["state"] = ["CA", "NY", "TX"]

    model = ssvm_reject_inference(x_labeled, y_labeled, x_unlabeled, n_neighbors=2)
    probabilities = model.predict_proba(x_unlabeled)[:, 1]

    assert probabilities.shape == (len(x_unlabeled),)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_mean_teacher_baseline_uses_soft_teacher_targets_when_student_has_sklearn_fit(reject_data):
    x_labeled, y_labeled, x_unlabeled = reject_data
    teacher = RecordingModel(probability_batches=[np.array([0.2, 0.8, 0.6])])
    student = RecordingModel()

    result = mean_teacher_baseline(student, teacher, x_labeled, y_labeled, x_unlabeled, n_iterations=1)

    assert result is student
    augmented = student.fit_calls[-1]
    assert len(augmented["X"]) == len(x_labeled) + 2 * len(x_unlabeled)
    assert augmented["y"].tolist() == [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]
    assert np.allclose(augmented["sample_weight"][: len(x_labeled)], 1.0)
    assert np.allclose(augmented["sample_weight"][len(x_labeled) : len(x_labeled) + len(x_unlabeled)], [0.4, 0.1, 0.2])
    assert np.allclose(augmented["sample_weight"][-len(x_unlabeled) :], [0.1, 0.4, 0.3])


def test_noisy_student_baseline_adds_teacher_hard_pseudo_labels(reject_data):
    x_labeled, y_labeled, x_unlabeled = reject_data
    teacher = RecordingModel(probability_batches=[np.array([0.2, 0.8, 0.6])])
    student = RecordingModel()

    result = noisy_student_baseline(
        student,
        teacher,
        x_labeled,
        y_labeled,
        x_unlabeled,
        noise_std=0.0,
        n_iterations=1,
    )

    assert result is student
    augmented = student.fit_calls[-1]
    assert len(augmented["X"]) == len(x_labeled) + len(x_unlabeled)
    assert augmented["y"].tolist() == [0, 0, 1, 1, 0, 1, 1]


def test_reject_inference_baselines_validate_inputs(reject_data):
    x_labeled, y_labeled, x_unlabeled = reject_data
    model = RecordingModel(probability_batches=[np.array([0.2, 0.7, 0.5])])

    with pytest.raises(ValueError, match="same length"):
        hard_augmentation(model, x_labeled, y_labeled[:-1], x_unlabeled)

    with pytest.raises(ValueError, match="threshold"):
        hard_augmentation(model, x_labeled, y_labeled, x_unlabeled, threshold=1.5)

    with pytest.raises(ValueError, match="n_bins"):
        parceling(model, x_labeled, y_labeled, x_unlabeled, n_bins=0)

    with pytest.raises(ValueError, match="confidence_threshold"):
        self_training(model, x_labeled, y_labeled, x_unlabeled, confidence_threshold=0.4)

    with pytest.raises(ValueError, match="eps"):
        ipw_weighted_pd(RecordingPropensity(np.ones(len(x_labeled))), model, x_labeled, y_labeled, eps=0)

    with pytest.raises(ValueError, match="quantile"):
        extrapolation_reject_inference(model, x_labeled, y_labeled, x_unlabeled, quantile=0)

    with pytest.raises(ValueError, match="n_epochs"):
        domain_adversarial_balancing(model, x_labeled, y_labeled, x_unlabeled, n_epochs=-1)

    with pytest.raises(ValueError, match="both classes"):
        ssvm_reject_inference(x_labeled, np.zeros(len(y_labeled), dtype=int), x_unlabeled)

    with pytest.raises(ValueError, match="ema_decay"):
        mean_teacher_baseline(model, model, x_labeled, y_labeled, x_unlabeled, ema_decay=1.5)

    with pytest.raises(ValueError, match="noise_std"):
        noisy_student_baseline(model, model, x_labeled, y_labeled, x_unlabeled, noise_std=-0.1)
