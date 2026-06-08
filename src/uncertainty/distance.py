from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.data.encoding import CategoricalStringifier


DEFAULT_DISTANCE_REFERENCE_ROWS = 2_000


def _build_distance_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in x_train.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("stringifier", CategoricalStringifier()),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def compute_knn_distance_uncertainty(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    k: int = 10,
    batch_size: int = 5_000,
    max_reference_rows: int | None = DEFAULT_DISTANCE_REFERENCE_ROWS,
    random_state: int = 42,
) -> np.ndarray:
    """Compute kNN-distance uncertainty using a sampled accepted reference set."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_reference_rows is not None and max_reference_rows <= 0:
        raise ValueError("max_reference_rows must be positive or None.")
    if len(x_train) == 0:
        raise ValueError("x_train must not be empty.")

    x_train = pd.DataFrame(x_train).copy()
    x_test = pd.DataFrame(x_test).copy()
    x_test = x_test.reindex(columns=x_train.columns)
    x_reference = _sample_reference_frame(x_train, max_reference_rows=max_reference_rows, random_state=random_state)

    preprocessor = _build_distance_preprocessor(x_reference)
    reference_scaled = _as_float_matrix(preprocessor.fit_transform(x_reference))
    test_scaled = _as_float_matrix(preprocessor.transform(x_test))

    n_neighbors = min(k, len(x_reference))
    reference_squared_norm = _sparse_squared_norm(reference_scaled)
    mean_distances = []
    for start in range(0, test_scaled.shape[0], batch_size):
        batch = test_scaled[start : start + batch_size]
        squared_distances = _sparse_pairwise_squared_distances(
            batch,
            reference_scaled,
            reference_squared_norm=reference_squared_norm,
        )
        nearest_squared = np.partition(squared_distances, n_neighbors - 1, axis=1)[:, :n_neighbors]
        mean_distances.append(np.sqrt(nearest_squared).mean(axis=1))
    return np.concatenate(mean_distances)


def _sample_reference_frame(
    frame: pd.DataFrame,
    max_reference_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    if max_reference_rows is None or len(frame) <= max_reference_rows:
        return frame.copy().reset_index(drop=True)
    return frame.sample(n=max_reference_rows, random_state=random_state).reset_index(drop=True)


def _as_float_matrix(matrix):
    if sparse.issparse(matrix):
        return matrix.astype(float).tocsr()
    return sparse.csr_matrix(np.asarray(matrix, dtype=float))


def _sparse_squared_norm(matrix) -> np.ndarray:
    matrix = matrix.astype(float).tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix, dtype=float)
    return np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()


def _sparse_pairwise_squared_distances(
    batch,
    reference,
    reference_squared_norm: np.ndarray | None = None,
) -> np.ndarray:
    batch = batch.astype(float).tocsr() if sparse.issparse(batch) else sparse.csr_matrix(batch, dtype=float)
    reference = (
        reference.astype(float).tocsr() if sparse.issparse(reference) else sparse.csr_matrix(reference, dtype=float)
    )
    if batch.shape[1] != reference.shape[1]:
        raise ValueError("batch and reference must have the same number of columns.")

    batch_squared_norm = _sparse_squared_norm(batch)
    if reference_squared_norm is None:
        reference_squared_norm = _sparse_squared_norm(reference)
    similarities = batch @ reference.T
    similarities = similarities.toarray() if sparse.issparse(similarities) else np.asarray(similarities, dtype=float)
    squared_distances = batch_squared_norm[:, None] + reference_squared_norm[None, :] - 2.0 * similarities
    return np.maximum(squared_distances, 0.0)


def _pairwise_squared_distances(batch: np.ndarray, reference: np.ndarray, feature_batch_size: int = 32) -> np.ndarray:
    batch = np.asarray(batch, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if batch.ndim != 2 or reference.ndim != 2:
        raise ValueError("batch and reference must be two-dimensional arrays.")
    if batch.shape[1] != reference.shape[1]:
        raise ValueError("batch and reference must have the same number of columns.")
    if feature_batch_size <= 0:
        raise ValueError("feature_batch_size must be positive.")

    squared_distances = np.zeros((len(batch), len(reference)), dtype=float)
    for start in range(0, batch.shape[1], feature_batch_size):
        end = min(start + feature_batch_size, batch.shape[1])
        difference = batch[:, None, start:end] - reference[None, :, start:end]
        squared_distances += np.square(difference).sum(axis=2)
    return squared_distances


def normalize_distance_against_reference(
    train_distances: np.ndarray,
    test_distances: np.ndarray,
) -> np.ndarray:
    """Quantile-normalize test distances against an accepted reference distribution."""
    train_distances = np.asarray(train_distances, dtype=float)
    test_distances = np.asarray(test_distances, dtype=float)
    if len(train_distances) == 0:
        raise ValueError("train_distances must not be empty.")

    sorted_reference = np.sort(train_distances)
    ranks = np.searchsorted(sorted_reference, test_distances, side="right")
    normalized = ranks / len(sorted_reference)
    return np.clip(normalized, 0.0, 1.0)
