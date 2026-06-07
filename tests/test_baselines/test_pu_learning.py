import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src.baselines.pu_learning import (
    PU_BASELINES,
    elkan_noto_correction,
    nnpu_loss,
    pu_bagging,
    pu_bagging_predict,
    upu_loss,
)


class RecordingPUModel(BaseEstimator, ClassifierMixin):
    def __init__(self, probabilities=None):
        self.probabilities = probabilities

    def fit(self, X, y):
        self.fit_X_ = pd.DataFrame(X).reset_index(drop=True)
        self.fit_y_ = np.asarray(y).copy()
        return self

    def predict_proba(self, X):
        n_rows = len(pd.DataFrame(X))
        if self.probabilities is None:
            probabilities = np.linspace(0.2, 0.8, n_rows)
        else:
            probabilities = np.resize(np.asarray(self.probabilities, dtype=float), n_rows)
        return np.column_stack([1.0 - probabilities, probabilities])


class IndexAwarePUModel(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        frame = pd.DataFrame(X).reset_index(drop=True)
        self.fit_X_ = frame
        self.fit_y_ = np.asarray(y).copy()
        self.offset_ = float(frame["score"].mean()) / 100.0
        return self

    def predict_proba(self, X):
        score = pd.DataFrame(X)["score"].to_numpy(dtype=float)
        probabilities = np.clip(0.1 + self.offset_ + score / 100.0, 0.0, 1.0)
        return np.column_stack([1.0 - probabilities, probabilities])


@pytest.fixture
def pu_data():
    x_labeled = pd.DataFrame({"score": [10, 20, 80, 90], "dti": [12, 14, 28, 30]})
    y_labeled = np.array([0, 0, 1, 1])
    x_unlabeled = pd.DataFrame({"score": [15, 35, 55, 75, 95], "dti": [13, 16, 22, 26, 32]})
    return x_labeled, y_labeled, x_unlabeled


def test_pu_baseline_registry_contains_task18_entries():
    expected = {"elkan_noto", "pu_bagging", "upu_loss", "nnpu_loss"}

    assert expected.issubset(PU_BASELINES)
    assert all(callable(builder) for builder in PU_BASELINES.values())


def test_elkan_noto_correction_scales_unlabeled_probabilities(pu_data):
    x_labeled, y_labeled, x_unlabeled = pu_data
    model = RecordingPUModel(probabilities=[0.2, 0.4, 0.5, 0.8, 0.9])

    fitted, corrected = elkan_noto_correction(model, x_labeled, y_labeled, x_unlabeled)

    assert fitted is model
    assert model.fit_y_.tolist() == [0, 0, 1, 1]
    assert corrected.shape == (len(x_unlabeled),)
    assert np.all((corrected >= 0) & (corrected <= 1))
    assert corrected[-1] == pytest.approx(1.0)


def test_pu_bagging_trains_cloned_models_with_sampled_unlabeled_negatives(pu_data):
    x_labeled, y_labeled, x_unlabeled = pu_data
    x_positive = x_labeled[y_labeled == 1].reset_index(drop=True)
    base_model = IndexAwarePUModel()

    models = pu_bagging(base_model, x_positive, x_unlabeled, n_bags=4, bag_size_ratio=0.4, random_state=7)

    assert len(models) == 4
    assert all(model is not base_model for model in models)
    assert all(len(model.fit_X_) == len(x_positive) + 2 for model in models)
    assert all(model.fit_y_.tolist() == [1, 1, 0, 0] for model in models)


def test_pu_bagging_predict_averages_member_predictions(pu_data):
    _, _, x_unlabeled = pu_data
    models = [RecordingPUModel(probabilities=[0.2, 0.4]), RecordingPUModel(probabilities=[0.6, 0.8])]

    predictions = pu_bagging_predict(models, x_unlabeled.head(2))

    assert np.allclose(predictions, [0.4, 0.6])


def test_upu_and_nnpu_losses_are_finite_and_nnpu_is_non_negative():
    y_pred = np.array([0.9, 0.8, 0.2, 0.3])
    y_true = np.array([1, 1, 0, 0])

    unbiased = upu_loss(y_pred, y_true, pi_p=0.25)
    non_negative = nnpu_loss(y_pred, y_true, pi_p=0.25)

    assert np.isfinite(unbiased)
    assert non_negative >= 0.0
    assert non_negative == max(0.0, unbiased)


def test_pu_functions_validate_inputs(pu_data):
    x_labeled, y_labeled, x_unlabeled = pu_data
    model = RecordingPUModel()

    with pytest.raises(ValueError, match="same length"):
        elkan_noto_correction(model, x_labeled, y_labeled[:-1], x_unlabeled)

    with pytest.raises(ValueError, match="positive"):
        elkan_noto_correction(model, x_labeled, np.zeros(len(y_labeled), dtype=int), x_unlabeled)

    with pytest.raises(ValueError, match="n_bags"):
        pu_bagging(model, x_labeled[y_labeled == 1], x_unlabeled, n_bags=0)

    with pytest.raises(ValueError, match="bag_size_ratio"):
        pu_bagging(model, x_labeled[y_labeled == 1], x_unlabeled, bag_size_ratio=0)

    with pytest.raises(ValueError, match="models"):
        pu_bagging_predict([], x_unlabeled)

    with pytest.raises(ValueError, match="pi_p"):
        nnpu_loss(np.array([0.2, 0.8]), np.array([0, 1]), pi_p=1.5)
