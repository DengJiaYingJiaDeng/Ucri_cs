from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.baselines.traditional import (
    DEEP_TABULAR_BASELINES,
    SUPPLEMENTARY_BASELINES,
    TRADITIONAL_BASELINES,
    build_catboost,
    build_ft_transformer,
    build_lightgbm,
    build_logistic_regression,
    build_mlp_focal,
    build_random_forest,
    build_saint,
    build_smote_baseline,
    build_tabnet,
    build_xgboost,
)


@pytest.fixture
def baseline_data():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(
        {
            "loan_amount": rng.lognormal(9, 0.4, 80),
            "dti": rng.uniform(5, 35, 80),
            "fico_avg": rng.normal(680, 35, 80),
        }
    )
    logit = -1.2 + 0.04 * x["dti"] - 0.003 * x["fico_avg"] + 0.08 * np.log(x["loan_amount"])
    probability = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, probability)
    return x, y


def test_traditional_baseline_registry_contains_required_models():
    expected = {
        "LogisticRegression",
        "RandomForest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "MLP",
        "FT-Transformer",
        "TabNet",
        "SAINT",
        "MLP-Focal",
        "LightGBM-SMOTE",
    }

    assert expected.issubset(TRADITIONAL_BASELINES)
    assert SUPPLEMENTARY_BASELINES == {"MLP-Focal", "LightGBM-SMOTE"}
    assert DEEP_TABULAR_BASELINES == {"FT-Transformer", "TabNet", "SAINT"}
    assert all(callable(builder) for builder in TRADITIONAL_BASELINES.values())


@pytest.mark.parametrize(
    "builder",
    [
        build_logistic_regression,
        build_random_forest,
        build_lightgbm,
        build_catboost,
        build_xgboost,
    ],
)
def test_core_traditional_baselines_fit_predict_proba(builder, baseline_data):
    x, y = baseline_data
    model = builder(random_state=7)

    model.fit(x, y)
    probabilities = model.predict_proba(x)[:, 1]

    assert probabilities.shape == (len(x),)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_xgboost_uses_sklearn_fallback_when_package_missing():
    model = build_xgboost(random_state=7)

    if find_spec("xgboost") is None:
        assert isinstance(model, GradientBoostingClassifier)
    else:
        assert model.__class__.__name__ == "XGBClassifier"


def test_optional_deep_baselines_return_estimators():
    builders = [build_ft_transformer, build_tabnet, build_saint, build_mlp_focal]

    for builder in builders:
        assert isinstance(builder(random_state=7), BaseEstimator)


def test_smote_baseline_returns_pipeline_or_lightgbm_fallback():
    model = build_smote_baseline(random_state=7)

    if isinstance(model, Pipeline):
        assert isinstance(model, Pipeline)
        assert list(model.named_steps) == ["smote", "classifier"]
    else:
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_proba")
