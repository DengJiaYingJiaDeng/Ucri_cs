import pickle
import warnings
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError

from src.models.student import StudentModel
from src.models.sklearn_compat import predict_proba_silencing_lightgbm_feature_name_warning


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


def test_student_soft_targets_expand_to_weighted_binary_bce(student_data):
    x, _, _, _, _ = student_data
    model = StudentModel(model_type="logistic")
    soft_targets = np.array([0.25, 0.8])
    sample_weights = np.array([2.0, 3.0])

    x_expanded, y_expanded, weights_expanded = model._expand_soft_targets(x.head(2), soft_targets, sample_weights)

    assert len(x_expanded) == 4
    assert np.array_equal(y_expanded, np.array([1, 1, 0, 0]))
    assert np.allclose(weights_expanded, np.array([0.5, 2.4, 1.5, 0.6]))


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


def test_student_handles_mixed_type_categorical_values(student_data):
    x, y, _, _, _ = student_data
    x = x.copy()
    x["state"] = ["CA", 1.0, np.nan, "TX"] * (len(x) // 4)
    model = StudentModel(model_type="logistic")

    model.fit(x, y)
    preds = model.predict_proba(x.head(20))

    assert preds.shape == (20,)


def test_student_logistic_preprocessor_keeps_high_cardinality_one_hot_sparse():
    x = pd.DataFrame(
        {
            "loan_amount": np.arange(40, dtype=float),
            "zip_code": [f"zip_{index}" for index in range(40)],
        }
    )
    model = StudentModel(model_type="logistic")

    transformed = model._build_preprocessor(x).fit_transform(x)

    assert sparse.issparse(transformed)


def test_student_catboost_preprocessor_uses_low_dimensional_categorical_codes():
    x = pd.DataFrame(
        {
            "loan_amount": np.arange(40, dtype=float),
            "zip_code": [f"zip_{index}" for index in range(40)],
        }
    )
    model = StudentModel(model_type="catboost")

    transformed = model._build_preprocessor(x).fit_transform(x)

    assert not sparse.issparse(transformed)
    assert transformed.shape[1] == 2


def test_student_forwards_gpu_params_to_lightgbm_estimator():
    if find_spec("lightgbm") is None:
        pytest.skip("LightGBM is not installed.")

    model = StudentModel(model_type="lightgbm", device_type="gpu", gpu_device_id=0)
    estimator = model._build_estimator(pos_weight=1.0)

    assert estimator.get_params()["device_type"] == "gpu"
    assert estimator.get_params()["gpu_device_id"] == 0


def test_student_lightgbm_predict_silences_pipeline_feature_name_warning(student_data):
    if find_spec("lightgbm") is None:
        pytest.skip("LightGBM is not installed.")

    x, y, _, _, _ = student_data
    model = StudentModel(model_type="lightgbm")
    model.fit(x, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict_proba(x.head(10))

    assert not any("X does not have valid feature names" in str(warning.message) for warning in caught)


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

    preds_after = predict_proba_silencing_lightgbm_feature_name_warning(loaded_model, x)[:, 1]
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
