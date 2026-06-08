import warnings
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError

from src.models.propensity import PropensityModel


@pytest.fixture
def prop_data():
    rng = np.random.default_rng(42)
    n = 500
    x = pd.DataFrame(
        {
            "loan_amount": rng.lognormal(9, 0.5, n),
            "dti": rng.uniform(5, 40, n),
            "emp_length": rng.integers(0, 30, n),
        }
    )
    logit = -2 + 0.5 * np.log(x["loan_amount"]) - 0.03 * x["dti"] + 0.05 * x["emp_length"]
    prob = 1 / (1 + np.exp(-logit))
    accepted = rng.binomial(1, prob)
    return x, accepted


def test_propensity_model_fit_predict(prop_data):
    x, accepted = prop_data
    model = PropensityModel(model_type="logistic")

    model.fit(x, accepted)
    probas = model.predict_proba(x)

    assert probas.shape == (len(x),)
    assert np.all((probas >= 0.01) & (probas <= 0.99))


def test_propensity_model_returns_clipped_probs(prop_data):
    x, accepted = prop_data
    model = PropensityModel(model_type="logistic")

    model.fit(x, accepted)
    probas = model.predict_proba(x)

    assert np.min(probas) >= 0.01
    assert np.max(probas) <= 0.99


def test_propensity_model_with_lightgbm(prop_data):
    x, accepted = prop_data
    model = PropensityModel(model_type="lightgbm")

    model.fit(x, accepted)
    probas = model.predict_proba(x)

    assert probas.shape == (len(x),)
    assert np.all((probas >= 0.01) & (probas <= 0.99))


def test_propensity_model_with_catboost(prop_data):
    x, accepted = prop_data
    model = PropensityModel(model_type="catboost")

    model.fit(x, accepted)
    probas = model.predict_proba(x)

    assert probas.shape == (len(x),)
    assert np.all((probas >= 0.01) & (probas <= 0.99))


def test_propensity_model_handles_categorical_shared_features(prop_data):
    x, accepted = prop_data
    x = x.copy()
    x["state"] = ["CA", "TX"] * (len(x) // 2)
    x["loan_purpose"] = ["debt_consolidation", "business", "credit_card", "other"] * (len(x) // 4)
    model = PropensityModel(model_type="logistic")

    model.fit(x, accepted)
    probas = model.predict_proba(x.head(10))

    assert probas.shape == (10,)


def test_propensity_model_handles_mixed_type_categorical_values(prop_data):
    x, accepted = prop_data
    x = x.copy()
    x["state"] = ["CA", 1.0, np.nan, "TX"] * (len(x) // 4)
    model = PropensityModel(model_type="logistic")

    model.fit(x, accepted)
    probas = model.predict_proba(x.head(10))

    assert probas.shape == (10,)


def test_propensity_logistic_preprocessor_keeps_high_cardinality_one_hot_sparse():
    x = pd.DataFrame(
        {
            "loan_amount": np.arange(40, dtype=float),
            "zip_code": [f"zip_{index}" for index in range(40)],
        }
    )
    model = PropensityModel(model_type="logistic")

    transformed = model._build_preprocessor(x).fit_transform(x)

    assert sparse.issparse(transformed)


def test_propensity_catboost_preprocessor_uses_low_dimensional_categorical_codes():
    x = pd.DataFrame(
        {
            "loan_amount": np.arange(40, dtype=float),
            "zip_code": [f"zip_{index}" for index in range(40)],
        }
    )
    model = PropensityModel(model_type="catboost")

    transformed = model._build_preprocessor(x).fit_transform(x)

    assert not sparse.issparse(transformed)
    assert transformed.shape[1] == 2


def test_propensity_lightgbm_predict_silences_pipeline_feature_name_warning(prop_data):
    if find_spec("lightgbm") is None:
        pytest.skip("LightGBM is not installed.")

    x, accepted = prop_data
    model = PropensityModel(model_type="lightgbm")
    model.fit(x, accepted)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict_proba(x.head(10))

    assert not any("X does not have valid feature names" in str(warning.message) for warning in caught)


def test_compute_weights_uses_clipped_inverse_propensity(prop_data):
    x, accepted = prop_data
    model = PropensityModel(model_type="logistic")

    model.fit(x, accepted)
    weights = model.compute_weights(x, eps=0.05)

    assert weights.shape == (len(x),)
    assert np.all(weights <= 20.0)
    assert np.all(weights >= 1.0)


def test_predict_before_fit_raises(prop_data):
    x, _ = prop_data
    model = PropensityModel(model_type="logistic")

    with pytest.raises(NotFittedError):
        model.predict_proba(x)


def test_unknown_propensity_model_type_raises(prop_data):
    x, accepted = prop_data
    model = PropensityModel(model_type="not_a_model")

    with pytest.raises(ValueError, match="Unknown propensity model type"):
        model.fit(x, accepted)
