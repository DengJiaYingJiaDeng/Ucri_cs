import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.uncertainty.distance import (
    _build_distance_preprocessor,
    _pairwise_squared_distances,
    compute_knn_distance_uncertainty,
    normalize_distance_against_reference,
)


@pytest.fixture
def distance_data():
    rng = np.random.default_rng(42)
    n_train = 200
    x_train = pd.DataFrame(
        {
            "loan_amount": rng.lognormal(9, 0.5, n_train),
            "dti": rng.uniform(5, 40, n_train),
            "emp_length": rng.integers(0, 30, n_train),
        }
    )
    n_test = 100
    x_test = pd.DataFrame(
        {
            "loan_amount": rng.lognormal(9, 0.5, n_test),
            "dti": rng.uniform(5, 40, n_test),
            "emp_length": rng.integers(0, 30, n_test),
        }
    )
    return x_train, x_test


def test_knn_distance_returns_non_negative(distance_data):
    x_train, x_test = distance_data

    result = compute_knn_distance_uncertainty(x_train, x_test, k=10)

    assert np.all(result >= 0)


def test_knn_distance_increases_for_ood_samples(distance_data):
    x_train, x_test = distance_data

    near_result = compute_knn_distance_uncertainty(x_train, x_test, k=10)
    x_ood = x_test * 3.0
    far_result = compute_knn_distance_uncertainty(x_train, x_ood, k=10)

    assert np.mean(far_result) > np.mean(near_result)


def test_knn_distance_shape_matches_input(distance_data):
    x_train, x_test = distance_data

    result = compute_knn_distance_uncertainty(x_train, x_test, k=10)

    assert len(result) == len(x_test)


def test_knn_distance_supports_categorical_features(distance_data):
    x_train, x_test = distance_data
    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train["state"] = ["CA", "TX", "NY", "FL"] * 50
    x_test["state"] = ["CA", "TX"] * 50

    result = compute_knn_distance_uncertainty(x_train, x_test, k=10)

    assert len(result) == len(x_test)
    assert np.all(np.isfinite(result))


def test_knn_distance_supports_mixed_type_categorical_values(distance_data):
    x_train, x_test = distance_data
    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train["state"] = ["CA", 1.0, np.nan, "TX"] * 50
    x_test["state"] = ["CA", np.nan, 1.0, "WA"] * 25

    result = compute_knn_distance_uncertainty(x_train, x_test, k=10)

    assert len(result) == len(x_test)
    assert np.all(np.isfinite(result))


def test_distance_preprocessor_keeps_high_cardinality_one_hot_sparse():
    x_train = pd.DataFrame(
        {
            "loan_amount": np.arange(40, dtype=float),
            "zip_code": [f"zip_{index}" for index in range(40)],
        }
    )

    transformed = _build_distance_preprocessor(x_train).fit_transform(x_train)

    assert sparse.issparse(transformed)


def test_knn_distance_supports_sampled_reference_rows(distance_data):
    x_train, x_test = distance_data

    result = compute_knn_distance_uncertainty(
        x_train,
        x_test.head(12),
        k=10,
        max_reference_rows=7,
        random_state=42,
    )

    assert len(result) == 12
    assert np.all(np.isfinite(result))


def test_pairwise_squared_distances_match_direct_broadcast():
    batch = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, -1.0]])
    reference = np.array([[1.0, 1.0, 1.0], [0.0, 2.0, 4.0], [5.0, 1.0, -2.0]])

    expected = ((batch[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
    result = _pairwise_squared_distances(batch, reference)

    assert np.allclose(result, expected)


def test_knn_distance_caps_k_to_training_size(distance_data):
    x_train, x_test = distance_data

    result = compute_knn_distance_uncertainty(x_train.head(5), x_test.head(3), k=10)

    assert len(result) == 3
    assert np.all(result >= 0)


def test_knn_distance_invalid_k_raises(distance_data):
    x_train, x_test = distance_data

    with pytest.raises(ValueError, match="k must be positive"):
        compute_knn_distance_uncertainty(x_train, x_test, k=0)

    with pytest.raises(ValueError, match="max_reference_rows"):
        compute_knn_distance_uncertainty(x_train, x_test, max_reference_rows=0)


def test_normalize_distance_against_reference_returns_quantiles():
    train_distances = np.array([0.1, 0.2, 0.3, 0.4])
    test_distances = np.array([0.05, 0.2, 0.5])

    normalized = normalize_distance_against_reference(train_distances, test_distances)

    assert normalized.tolist() == [0.0, 0.5, 1.0]
    assert np.all((normalized >= 0) & (normalized <= 1))


def test_normalize_distance_empty_reference_raises():
    with pytest.raises(ValueError, match="train_distances"):
        normalize_distance_against_reference(np.array([]), np.array([0.1]))
