import numpy as np
import pandas as pd
import pytest

from src.baselines.reject_inference import (
    REJECT_INFERENCE_BASELINES,
    fuzzy_augmentation,
    hard_augmentation,
    ipw_weighted_pd,
    parceling,
    self_training,
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


def test_reject_inference_registry_contains_task17_baselines():
    expected = {"hard", "fuzzy", "parceling", "self_training", "ipw"}

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
