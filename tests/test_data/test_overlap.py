import numpy as np
import pandas as pd
import pytest

from src.data.overlap import overlap_diagnostics, overlap_filter, overlap_k_sensitivity


def test_overlap_filter_returns_boolean_mask():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 500),
            "b": rng.normal(0, 1, 500),
        }
    )
    propensities = rng.uniform(0.05, 0.95, 500)

    mask = overlap_filter(X, X, propensities)

    assert mask.dtype == bool
    assert len(mask) == 500


def test_overlap_filter_flags_extreme_propensity():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"a": rng.normal(0, 1, 100)})
    propensities = np.array([0.02] * 50 + [0.5] * 50)

    diagnostics = overlap_diagnostics(X, X, propensities, epsilon_low=0.05)

    assert not diagnostics["propensity_mask"][:50].all()
    assert not diagnostics["in_overlap_mask"][:50].all()


def test_overlap_filter_flags_ood_numeric_features():
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame({"a": rng.normal(0, 1, 500)})
    X_test = pd.DataFrame({"a": rng.normal(5, 1, 100)})
    propensities = np.full(100, 0.5)

    diagnostics = overlap_diagnostics(X_train, X_test, propensities)

    assert not diagnostics["in_overlap_mask"].all()
    assert not diagnostics["distance_mask"].all()
    assert not diagnostics["range_mask"].all()
    assert diagnostics["out_of_support_rate"] > 0.5


def test_overlap_filter_flags_unseen_categories():
    X_train = pd.DataFrame(
        {
            "amount": [1000, 1200, 1400, 1600, 1800, 2000],
            "state": ["CA", "NY", "CA", "TX", "NY", "TX"],
        }
    )
    X_test = pd.DataFrame(
        {
            "amount": [1300, 1500, 1700],
            "state": ["CA", "FL", "NY"],
        }
    )
    propensities = np.full(len(X_test), 0.5)

    diagnostics = overlap_diagnostics(
        X_train,
        X_test,
        propensities,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )

    assert diagnostics["category_mask"].tolist() == [True, False, True]
    assert diagnostics["in_overlap_mask"].tolist() == [True, False, True]


def test_overlap_filter_supports_application_dates():
    X_train = pd.DataFrame(
        {
            "application_date": pd.date_range("2014-01-01", periods=30, freq="MS").astype(str),
            "loan_amount": np.linspace(1000, 5000, 30),
        }
    )
    X_test = pd.DataFrame(
        {
            "application_date": ["2014-06-01", "2030-01-01"],
            "loan_amount": [2500, 3000],
        }
    )
    propensities = np.full(2, 0.5)

    diagnostics = overlap_diagnostics(X_train, X_test, propensities)

    assert diagnostics["range_mask"].tolist() == [True, False]
    assert diagnostics["in_overlap_mask"].tolist() == [True, False]


def test_overlap_k_sensitivity_reports_coverage_for_each_k():
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame({"a": rng.normal(0, 1, 80), "b": rng.normal(0, 1, 80)})
    X_test = pd.DataFrame({"a": rng.normal(0, 1, 25), "b": rng.normal(0, 1, 25)})
    propensities = np.full(len(X_test), 0.5)

    result = overlap_k_sensitivity(X_train, X_test, propensities, k_values=[3, 5])

    assert set(result) == {"k=3", "k=5"}
    for entry in result.values():
        assert 0.0 <= entry["coverage"] <= 1.0
        assert entry["n_in_overlap"] + entry["n_out_of_support"] == len(X_test)


def test_overlap_filter_validates_inputs():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="same length"):
        overlap_filter(X, X, np.array([0.5, 0.5]))

    with pytest.raises(ValueError, match="propensities"):
        overlap_filter(X, X, np.array([0.5, 1.2, 0.5]))

    with pytest.raises(ValueError, match="epsilon_low"):
        overlap_filter(X, X, np.array([0.5, 0.5, 0.5]), epsilon_low=-0.1)

    with pytest.raises(ValueError, match="k"):
        overlap_filter(X, X, np.array([0.5, 0.5, 0.5]), k=0)
