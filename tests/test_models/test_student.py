import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.models.student import StudentModel, _soft_bce_grad_hess


@pytest.fixture
def student_data():
    rng = np.random.default_rng(42)
    n_labeled = 320
    x_labeled = pd.DataFrame(
        {
            "loan_amount": rng.lognormal(9, 0.5, n_labeled),
            "dti": rng.uniform(5, 40, n_labeled),
            "emp_length": rng.integers(0, 30, n_labeled),
            "fico_avg": rng.normal(680, 45, n_labeled),
        }
    )
    logit = -3.0 + 0.25 * np.log(x_labeled["loan_amount"]) + 0.035 * x_labeled["dti"] - 0.004 * x_labeled["fico_avg"]
    prob = 1 / (1 + np.exp(-logit))
    y_labeled = rng.binomial(1, prob)

    n_rejected = 120
    x_rejected = pd.DataFrame(
        {
            "loan_amount": rng.lognormal(9.2, 0.6, n_rejected),
            "dti": rng.uniform(10, 45, n_rejected),
            "emp_length": rng.integers(0, 25, n_rejected),
            "fico_avg": rng.normal(660, 55, n_rejected),
        }
    )
    teacher_probs = rng.beta(3, 7, n_rejected)
    weights = rng.uniform(0.05, 1.0, n_rejected)
    return x_labeled, y_labeled, x_rejected, teacher_probs, weights


def test_soft_bce_grad_hess_uses_soft_targets():
    pred = np.array([0.2, 0.7])
    target = np.array([0.4, 0.6])

    grad, hess = _soft_bce_grad_hess(pred, target)

    assert np.allclose(grad, [-0.2, 0.1])
    assert np.all(hess > 0)


def test_student_fit_supervised_only(student_data):
    x, y, _, _, _ = student_data
    model = StudentModel(model_type="lightgbm")

    model.fit(x, y)
    preds = model.predict_proba(x)

    assert preds.shape == (len(x),)
    assert np.all((preds >= 0) & (preds <= 1))


def test_student_fit_with_soft_distillation(student_data):
    x, y, x_rej, teacher_probs, weights = student_data
    model = StudentModel(model_type="lightgbm")

    model.fit(x, y, x_rej, teacher_probs, weights, lambda_distill=1.0)
    preds = model.predict_proba(x)

    assert preds.shape == (len(x),)
    assert np.all((preds >= 0) & (preds <= 1))
    assert np.any((model.training_targets_ > 0) & (model.training_targets_ < 1))


def test_student_soft_labels_are_not_thresholded(student_data):
    x, y, x_rej, teacher_probs, weights = student_data
    model = StudentModel(model_type="logistic")

    model.fit(x, y, x_rej, teacher_probs, weights, lambda_distill=0.5)

    stored_rejected_targets = model.training_targets_[len(x) :]
    assert np.allclose(stored_rejected_targets, np.clip(teacher_probs, 1e-6, 1 - 1e-6))
    assert not np.all(np.isin(stored_rejected_targets, [0.0, 1.0]))


def test_student_post_calibration(student_data):
    x, y, _, _, _ = student_data
    model = StudentModel(model_type="lightgbm")
    model.fit(x, y)

    report = model.post_calibrate(x, y)
    preds = model.predict_proba(x)

    assert {"before_ECE", "after_ECE", "temperature"}.issubset(report)
    assert report["temperature"] > 0
    assert preds.shape == (len(x),)
    assert np.all((preds >= 0) & (preds <= 1))


def test_student_class_weight_applied(student_data):
    x, y, _, _, _ = student_data
    y_imbalanced = np.zeros(len(y), dtype=int)
    y_imbalanced[: len(y) // 10] = 1
    model = StudentModel(model_type="lightgbm")

    model.fit(x, y_imbalanced)

    assert model.scale_pos_weight > 1.0
    assert model.scale_pos_weight <= 20.0


def test_student_handles_categorical_features(student_data):
    x, y, _, _, _ = student_data
    x = x.copy()
    x["state"] = ["CA", "TX"] * (len(x) // 2)
    model = StudentModel(model_type="logistic")

    model.fit(x, y)
    preds = model.predict_proba(x.head(20))

    assert preds.shape == (20,)


def test_student_save_load_model_pipeline(student_data, tmp_path):
    x, y, _, _, _ = student_data
    model = StudentModel(model_type="lightgbm")
    model.fit(x, y)
    preds_before = model.predict_proba(x)
    path = tmp_path / "student_pipeline.pkl"

    with path.open("wb") as handle:
        pickle.dump(model.model, handle)
    with path.open("rb") as handle:
        loaded_model = pickle.load(handle)

    preds_after = loaded_model.predict_proba(x)[:, 1]
    assert np.allclose(preds_before, preds_after)


def test_tune_scale_pos_weight_returns_best_weight(student_data):
    x, y, _, _, _ = student_data

    report = StudentModel.tune_scale_pos_weight(
        x.iloc[:240],
        y[:240],
        x.iloc[240:],
        y[240:],
        model_type="logistic",
        cap_values=[5.0, 10.0],
    )

    assert report["best_pos_weight"] in {5.0, 10.0}
    assert len(report["tuning_results"]) == 2


def test_predict_before_fit_raises(student_data):
    x, _, _, _, _ = student_data
    model = StudentModel(model_type="lightgbm")

    with pytest.raises(NotFittedError):
        model.predict_proba(x)


def test_unknown_student_model_type_raises(student_data):
    x, y, _, _, _ = student_data
    model = StudentModel(model_type="unknown")

    with pytest.raises(ValueError, match="Unknown student model type"):
        model.fit(x, y)


def test_student_rejects_mismatched_distillation_lengths(student_data):
    x, y, x_rej, teacher_probs, weights = student_data
    model = StudentModel(model_type="lightgbm")

    with pytest.raises(ValueError, match="same length"):
        model.fit(x, y, x_rej, teacher_probs[:-1], weights)


def test_student_rejects_negative_pseudo_weights(student_data):
    x, y, x_rej, teacher_probs, weights = student_data
    weights = weights.copy()
    weights[0] = -1
    model = StudentModel(model_type="lightgbm")

    with pytest.raises(ValueError, match="pseudo_weights"):
        model.fit(x, y, x_rej, teacher_probs, weights)
