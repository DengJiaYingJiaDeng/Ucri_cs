from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


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
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def compute_knn_distance_uncertainty(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    k: int = 10,
    batch_size: int = 1024,
) -> np.ndarray:
    """Compute kNN-distance uncertainty using robust standardized features."""
    if k <= 0:
        raise ValueError("k must be positive.")
    if len(x_train) == 0:
        raise ValueError("x_train must not be empty.")

    x_train = pd.DataFrame(x_train).copy()
    x_test = pd.DataFrame(x_test).copy()
    x_test = x_test.reindex(columns=x_train.columns)

    preprocessor = _build_distance_preprocessor(x_train)
    train_scaled = np.asarray(preprocessor.fit_transform(x_train), dtype=float)
    test_scaled = np.asarray(preprocessor.transform(x_test), dtype=float)

    n_neighbors = min(k, len(x_train))
    mean_distances = []
    for start in range(0, len(test_scaled), batch_size):
        batch = test_scaled[start : start + batch_size]
        squared_distances = ((batch[:, None, :] - train_scaled[None, :, :]) ** 2).sum(axis=2)
        nearest_squared = np.partition(squared_distances, n_neighbors - 1, axis=1)[:, :n_neighbors]
        mean_distances.append(np.sqrt(nearest_squared).mean(axis=1))
    return np.concatenate(mean_distances)


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
