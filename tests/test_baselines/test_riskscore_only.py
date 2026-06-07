import numpy as np
import pandas as pd
import pytest

from src.baselines.riskscore_only import (
    RISKSCORE_ONLY_BASELINES,
    RiskScoreBinning,
    RiskScoreDtiLogisticRegression,
    RiskScoreIsotonicRegression,
    RiskScoreLogisticRegression,
    build_riskscore_only_models,
    fit_riskscore_dti_lr,
    fit_riskscore_lr,
)


@pytest.fixture
def riskscore_data():
    rng = np.random.default_rng(42)
    risk_score = np.linspace(520, 760, 120)
    dti = rng.uniform(5, 40, len(risk_score))
    probability = 1 / (1 + np.exp(-(-2.0 - 0.018 * (risk_score - 650) + 0.04 * dti)))
    y = rng.binomial(1, probability)
    x = pd.DataFrame({"risk_score": risk_score, "dti": dti})
    return x, y


def test_riskscore_only_registry_contains_required_models():
    expected = {"risk_score_binning", "risk_score_lr", "risk_score_dti_lr", "risk_score_isotonic"}

    assert expected == set(RISKSCORE_ONLY_BASELINES)
    assert expected == set(build_riskscore_only_models())
    assert all(callable(builder) for builder in RISKSCORE_ONLY_BASELINES.values())


@pytest.mark.parametrize(
    "model",
    [
        RiskScoreBinning(n_bins=8),
        RiskScoreLogisticRegression(random_state=7),
        RiskScoreDtiLogisticRegression(random_state=7),
        RiskScoreIsotonicRegression(),
    ],
)
def test_riskscore_baselines_fit_predict_proba(model, riskscore_data):
    x, y = riskscore_data

    model.fit(x, y)
    probabilities = model.predict_proba(x)

    assert probabilities.shape == (len(x), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_riskscore_binning_uses_training_edges_for_new_scores(riskscore_data):
    x, y = riskscore_data
    model = RiskScoreBinning(n_bins=5).fit(x, y)
    new_x = pd.DataFrame({"risk_score": [400, 600, 900]})

    probabilities = model.predict_proba(new_x)[:, 1]

    assert probabilities.shape == (3,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_riskscore_logistic_helpers_return_fitted_models(riskscore_data):
    x, y = riskscore_data

    risk_only = fit_riskscore_lr(x, y, random_state=7)
    risk_dti = fit_riskscore_dti_lr(x, y, random_state=7)

    assert risk_only.predict_proba(x).shape == (len(x), 2)
    assert risk_dti.predict_proba(x).shape == (len(x), 2)
    assert risk_dti.features_ == ["risk_score", "dti"]


def test_riskscore_baselines_accept_raw_risk_score_name(riskscore_data):
    x, y = riskscore_data
    raw_x = x.rename(columns={"risk_score": "Risk_Score"})

    model = RiskScoreLogisticRegression(random_state=7).fit(raw_x, y)

    assert model.predict_proba(raw_x).shape == (len(raw_x), 2)


def test_riskscore_baselines_validate_inputs(riskscore_data):
    x, y = riskscore_data

    with pytest.raises(KeyError, match="risk_score"):
        RiskScoreLogisticRegression().fit(x.drop(columns=["risk_score"]), y)

    with pytest.raises(ValueError, match="same length"):
        RiskScoreBinning().fit(x, y[:-1])

    with pytest.raises(ValueError, match="binary"):
        RiskScoreIsotonicRegression().fit(x, np.full(len(y), 2))
