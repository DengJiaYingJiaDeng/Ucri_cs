import warnings
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError

from src.models.teacher import TeacherEnsemble
from src.models.torch_mlp import TorchMLPClassifier


@pytest.fixture
def teacher_data():
    rng = np.random.default_rng(42)
    n = 500
    x = pd.DataFrame(
        {
            "loan_amount": rng.lognormal(9, 0.5, n),
            "dti": rng.uniform(5, 40, n),
            "emp_length": rng.integers(0, 30, n),
            "fico_avg": rng.normal(680, 40, n),
        }
    )
    logit = -1 + 0.3 * np.log(x["loan_amount"]) - 0.02 * x["dti"] - 0.01 * x["fico_avg"] / 100
    prob = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, prob)
    return x, y


def test_teacher_ensemble_fit_predict(teacher_data):
    x, y = teacher_data
    ensemble = TeacherEnsemble(n_models=3, model_types=["lightgbm", "catboost", "lightgbm"])

    ensemble.fit(x, y)
    preds = ensemble.predict_proba(x)

    assert preds.shape == (len(x),)
    assert np.all((preds >= 0) & (preds <= 1))


def test_teacher_ensemble_uncertainty(teacher_data):
    x, y = teacher_data
    ensemble = TeacherEnsemble(n_models=3)

    ensemble.fit(x, y)
    uncertainty = ensemble.compute_uncertainty(x)

    assert {"variance", "entropy", "margin", "mean"}.issubset(uncertainty)
    assert len(uncertainty["variance"]) == len(x)
    assert np.all(uncertainty["variance"] >= 0)
    assert np.all((uncertainty["entropy"] >= 0) & (uncertainty["margin"] >= 0))


def test_teacher_ensemble_calibration(teacher_data):
    x, y = teacher_data
    ensemble = TeacherEnsemble(n_models=3)

    ensemble.fit(x, y)
    ensemble.calibrate(x, y, method="temperature")
    calibrated = ensemble.predict_calibrated(x)

    assert ensemble.calibrated
    assert 0.01 <= ensemble.temperature <= 10.0
    assert calibrated.shape == (len(x),)
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_predict_calibrated_before_calibration_returns_mean_prediction(teacher_data):
    x, y = teacher_data
    ensemble = TeacherEnsemble(n_models=3)

    ensemble.fit(x, y)

    assert np.allclose(ensemble.predict_calibrated(x), ensemble.predict_proba(x))


def test_ensemble_disagreement_increases_away_from_data(teacher_data):
    x, y = teacher_data
    ensemble = TeacherEnsemble(n_models=3)

    ensemble.fit(x, y)
    x_far = x * 2.0
    unc_near = ensemble.compute_uncertainty(x)["variance"]
    unc_far = ensemble.compute_uncertainty(x_far)["variance"]

    assert np.mean(unc_far) >= np.mean(unc_near) * 0.4


def test_teacher_ensemble_handles_categorical_features(teacher_data):
    x, y = teacher_data
    x = x.copy()
    x["state"] = ["CA", "TX"] * (len(x) // 2)
    ensemble = TeacherEnsemble(n_models=2, model_types=["mlp", "lightgbm"])

    ensemble.fit(x, y)
    preds = ensemble.predict_proba(x.head(20))

    assert preds.shape == (20,)


def test_teacher_ensemble_handles_mixed_type_categorical_values(teacher_data):
    x, y = teacher_data
    x = x.copy()
    x["state"] = ["CA", 1.0, np.nan, "TX"] * (len(x) // 4)
    ensemble = TeacherEnsemble(n_models=1, model_types=["mlp"])

    ensemble.fit(x, y)
    preds = ensemble.predict_proba(x.head(20))

    assert preds.shape == (20,)


def test_teacher_lightgbm_preprocessor_keeps_high_cardinality_one_hot_sparse():
    x = pd.DataFrame(
        {
            "loan_amount": np.arange(40, dtype=float),
            "zip_code": [f"zip_{index}" for index in range(40)],
        }
    )
    ensemble = TeacherEnsemble(n_models=1, model_types=["lightgbm"])

    transformed = ensemble._build_preprocessor(x, "lightgbm").fit_transform(x)

    if find_spec("lightgbm") is not None:
        assert sparse.issparse(transformed)
    else:
        assert not sparse.issparse(transformed)
        assert transformed.shape[1] == 2


def test_teacher_catboost_preprocessor_uses_low_dimensional_categorical_codes():
    x = pd.DataFrame(
        {
            "loan_amount": np.arange(40, dtype=float),
            "zip_code": [f"zip_{index}" for index in range(40)],
        }
    )
    ensemble = TeacherEnsemble(n_models=1, model_types=["catboost"])

    transformed = ensemble._build_preprocessor(x, "catboost").fit_transform(x)

    assert not sparse.issparse(transformed)
    assert transformed.shape[1] == 2


def test_teacher_ensemble_forwards_gpu_params_to_optional_estimators():
    if find_spec("lightgbm") is None and find_spec("catboost") is None:
        pytest.skip("Optional gradient boosting libraries are not installed.")

    ensemble = TeacherEnsemble(n_models=1, model_types=["mlp"], device_type="gpu", gpu_device_id=0)

    if find_spec("lightgbm") is not None:
        lightgbm_estimator = ensemble._build_estimator("lightgbm", seed=42, pos_weight=1.0)
        lightgbm_params = lightgbm_estimator.get_params()
        assert lightgbm_params["device_type"] == "gpu"
        assert lightgbm_params["gpu_device_id"] == 0

    if find_spec("catboost") is not None:
        catboost_estimator = ensemble._build_estimator("catboost", seed=42, pos_weight=1.0)
        catboost_params = catboost_estimator.get_params()
        assert catboost_params["task_type"] == "GPU"
        assert catboost_params["devices"] == "0"


def test_teacher_mlp_branch_uses_real_neural_network_classifier():
    ensemble = TeacherEnsemble(n_models=1, model_types=["mlp"])

    estimator = ensemble._build_estimator("mlp", seed=42, pos_weight=1.0)

    assert isinstance(estimator, TorchMLPClassifier)


def test_teacher_lightgbm_predict_silences_pipeline_feature_name_warning(teacher_data):
    if find_spec("lightgbm") is None:
        pytest.skip("LightGBM is not installed.")

    x, y = teacher_data
    ensemble = TeacherEnsemble(n_models=1, model_types=["lightgbm"])
    ensemble.fit(x, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ensemble.predict_proba(x.head(10))

    assert not any("X does not have valid feature names" in str(warning.message) for warning in caught)


def test_pos_weight_is_capped_at_20():
    y = np.array([0] * 100 + [1])
    ensemble = TeacherEnsemble(n_models=1)

    assert ensemble._compute_pos_weight(y) == 20.0


def test_predict_before_fit_raises(teacher_data):
    x, _ = teacher_data
    ensemble = TeacherEnsemble(n_models=1)

    with pytest.raises(NotFittedError):
        ensemble.predict_proba(x)


def test_unknown_teacher_model_type_raises(teacher_data):
    x, y = teacher_data
    ensemble = TeacherEnsemble(n_models=1, model_types=["unknown"])

    with pytest.raises(ValueError, match="Unknown teacher model type"):
        ensemble.fit(x, y)


def test_model_type_count_must_match_n_models():
    with pytest.raises(ValueError, match="model_types length"):
        TeacherEnsemble(n_models=2, model_types=["lightgbm"])
