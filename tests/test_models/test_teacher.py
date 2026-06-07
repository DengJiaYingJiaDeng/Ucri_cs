import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.models.teacher import TeacherEnsemble


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

    assert np.mean(unc_far) >= np.mean(unc_near) * 0.5


def test_teacher_ensemble_handles_categorical_features(teacher_data):
    x, y = teacher_data
    x = x.copy()
    x["state"] = ["CA", "TX"] * (len(x) // 2)
    ensemble = TeacherEnsemble(n_models=2, model_types=["mlp", "lightgbm"])

    ensemble.fit(x, y)
    preds = ensemble.predict_proba(x.head(20))

    assert preds.shape == (20,)


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
