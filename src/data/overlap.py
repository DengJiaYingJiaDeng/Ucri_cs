from __future__ import annotations

import numpy as np
import pandas as pd


def overlap_filter(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    propensities: np.ndarray,
    epsilon_low: float = 0.05,
    epsilon_high: float = 0.05,
    k: int = 10,
    distance_quantile: float = 0.95,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> np.ndarray:
    """Return mask for samples inside empirical accepted-support overlap.

    A sample is in-overlap when its approval propensity is not extreme, its
    kNN distance to accepted training samples is within the accepted reference
    distribution, numeric/date features are inside winsorized training ranges,
    and categorical values were observed in training.
    """
    diagnostics = overlap_diagnostics(
        X_train,
        X_test,
        propensities,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        k=k,
        distance_quantile=distance_quantile,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    return diagnostics["in_overlap_mask"]


def overlap_diagnostics(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    propensities: np.ndarray,
    epsilon_low: float = 0.05,
    epsilon_high: float = 0.05,
    k: int = 10,
    distance_quantile: float = 0.95,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> dict[str, object]:
    """Return overlap mask components and coverage diagnostics."""
    x_train, x_test = _validate_frames(X_train, X_test)
    propensities = _validate_propensities(propensities, expected_length=len(x_test))
    _validate_thresholds(epsilon_low, epsilon_high, distance_quantile, lower_quantile, upper_quantile)
    if k <= 0:
        raise ValueError("k must be positive.")

    propensity_mask = (propensities >= epsilon_low) & (propensities <= (1.0 - epsilon_high))
    train_distances, test_distances = _mean_knn_distances(x_train, x_test, k=k)
    distance_threshold = float(np.quantile(train_distances, distance_quantile))
    distance_mask = test_distances <= distance_threshold

    range_mask = _feature_range_mask(x_train, x_test, lower_quantile, upper_quantile)
    category_mask = _category_overlap_mask(x_train, x_test)
    in_overlap = propensity_mask & distance_mask & range_mask & category_mask

    return {
        "in_overlap_mask": in_overlap,
        "propensity_mask": propensity_mask,
        "distance_mask": distance_mask,
        "range_mask": range_mask,
        "category_mask": category_mask,
        "knn_distance": test_distances,
        "distance_threshold": distance_threshold,
        "coverage": float(in_overlap.mean()) if len(in_overlap) else 0.0,
        "out_of_support_rate": float((~in_overlap).mean()) if len(in_overlap) else 0.0,
        "n_in_overlap": int(in_overlap.sum()),
        "n_out_of_support": int((~in_overlap).sum()),
        "n_samples": int(len(in_overlap)),
    }


def overlap_k_sensitivity(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    propensities: np.ndarray,
    k_values: list[int] | None = None,
    **kwargs,
) -> dict[str, dict[str, float | int]]:
    """Report overlap coverage sensitivity for k values such as {5, 10, 20}."""
    selected_k = k_values or [5, 10, 20]
    if len(selected_k) == 0:
        raise ValueError("k_values must not be empty.")

    results: dict[str, dict[str, float | int]] = {}
    for k in selected_k:
        diagnostics = overlap_diagnostics(X_train, X_test, propensities, k=k, **kwargs)
        results[f"k={k}"] = {
            "coverage": float(diagnostics["coverage"]),
            "out_of_support_rate": float(diagnostics["out_of_support_rate"]),
            "n_in_overlap": int(diagnostics["n_in_overlap"]),
            "n_out_of_support": int(diagnostics["n_out_of_support"]),
        }
    return results


def _validate_frames(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = pd.DataFrame(X_train).copy().reset_index(drop=True)
    x_test = pd.DataFrame(X_test).copy().reset_index(drop=True)
    if len(x_train) == 0:
        raise ValueError("X_train must not be empty.")
    if len(x_test) == 0:
        raise ValueError("X_test must not be empty.")
    if not x_train.columns.is_unique or not x_test.columns.is_unique:
        raise ValueError("Feature names must be unique.")
    x_test = x_test.reindex(columns=x_train.columns)
    return x_train, x_test


def _validate_propensities(propensities: np.ndarray, expected_length: int) -> np.ndarray:
    values = np.asarray(propensities, dtype=float)
    if values.ndim != 1:
        raise ValueError("propensities must be a one-dimensional array.")
    if len(values) != expected_length:
        raise ValueError("propensities must have the same length as X_test.")
    if not np.all(np.isfinite(values)):
        raise ValueError("propensities must contain finite values.")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("propensities must be in [0, 1].")
    return values


def _validate_thresholds(
    epsilon_low: float,
    epsilon_high: float,
    distance_quantile: float,
    lower_quantile: float,
    upper_quantile: float,
) -> None:
    if not 0.0 <= float(epsilon_low) < 1.0:
        raise ValueError("epsilon_low must be in [0, 1).")
    if not 0.0 <= float(epsilon_high) < 1.0:
        raise ValueError("epsilon_high must be in [0, 1).")
    if float(epsilon_low) + float(epsilon_high) >= 1.0:
        raise ValueError("epsilon_low + epsilon_high must be less than 1.")
    if not 0.0 < float(distance_quantile) <= 1.0:
        raise ValueError("distance_quantile must be in (0, 1].")
    if not 0.0 <= float(lower_quantile) < float(upper_quantile) <= 1.0:
        raise ValueError("lower_quantile and upper_quantile must satisfy 0 <= lower < upper <= 1.")


def _mean_knn_distances(X_train: pd.DataFrame, X_test: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    train_matrix, test_matrix = _preprocess_for_distance(X_train, X_test)
    n_neighbors = min(k, len(train_matrix))
    train_distances = _mean_nearest_distances(train_matrix, train_matrix, n_neighbors=n_neighbors)
    test_distances = _mean_nearest_distances(test_matrix, train_matrix, n_neighbors=n_neighbors)
    return train_distances, test_distances


def _preprocess_for_distance(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    numeric_columns = _numeric_or_date_columns(X_train)
    categorical_columns = [column for column in X_train.columns if column not in numeric_columns]

    matrices_train = []
    matrices_test = []
    if numeric_columns:
        train_numeric = _numeric_or_date_frame(X_train[numeric_columns])
        test_numeric = _numeric_or_date_frame(X_test[numeric_columns])
        medians = train_numeric.median(numeric_only=True).fillna(0.0)
        train_numeric = train_numeric.fillna(medians)
        test_numeric = test_numeric.fillna(medians)
        center = train_numeric.median()
        scale = (train_numeric.quantile(0.75) - train_numeric.quantile(0.25)).replace(0.0, np.nan).fillna(1.0)
        matrices_train.append(((train_numeric - center) / scale).to_numpy(dtype=float))
        matrices_test.append(((test_numeric - center) / scale).to_numpy(dtype=float))

    if categorical_columns:
        train_cat, test_cat = _one_hot_categoricals(X_train[categorical_columns], X_test[categorical_columns])
        matrices_train.append(train_cat)
        matrices_test.append(test_cat)

    if not matrices_train:
        raise ValueError("At least one feature column is required.")
    return np.hstack(matrices_train), np.hstack(matrices_test)


def _mean_nearest_distances(
    query: np.ndarray,
    reference: np.ndarray,
    n_neighbors: int,
    batch_size: int = 128,
    feature_batch_size: int = 8,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if feature_batch_size <= 0:
        raise ValueError("feature_batch_size must be positive.")

    results = []
    for start in range(0, len(query), batch_size):
        batch = query[start : start + batch_size]
        squared = _pairwise_squared_distances(batch, reference, feature_batch_size=feature_batch_size)
        nearest = np.partition(squared, n_neighbors - 1, axis=1)[:, :n_neighbors]
        results.append(np.sqrt(nearest).mean(axis=1))
    return np.concatenate(results)


def _pairwise_squared_distances(batch: np.ndarray, reference: np.ndarray, feature_batch_size: int = 8) -> np.ndarray:
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


def _feature_range_mask(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    lower_quantile: float,
    upper_quantile: float,
) -> np.ndarray:
    numeric_columns = _numeric_or_date_columns(X_train)
    if not numeric_columns:
        return np.ones(len(X_test), dtype=bool)

    train_numeric = _numeric_or_date_frame(X_train[numeric_columns])
    test_numeric = _numeric_or_date_frame(X_test[numeric_columns])
    lower = train_numeric.quantile(lower_quantile)
    upper = train_numeric.quantile(upper_quantile)

    mask = np.ones(len(X_test), dtype=bool)
    for column in numeric_columns:
        values = test_numeric[column]
        col_mask = values.isna() | ((values >= lower[column]) & (values <= upper[column]))
        mask &= col_mask.to_numpy(dtype=bool)
    return mask


def _category_overlap_mask(X_train: pd.DataFrame, X_test: pd.DataFrame) -> np.ndarray:
    categorical_columns = [column for column in X_train.columns if column not in _numeric_or_date_columns(X_train)]
    if not categorical_columns:
        return np.ones(len(X_test), dtype=bool)

    mask = np.ones(len(X_test), dtype=bool)
    for column in categorical_columns:
        seen = set(X_train[column].astype("string").fillna("missing").tolist())
        values = X_test[column].astype("string").fillna("missing")
        mask &= values.isin(seen).to_numpy(dtype=bool)
    return mask


def _numeric_or_date_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            columns.append(column)
        elif "date" in str(column).lower() or column in {"issue_d", "application_date"}:
            parsed = _parse_dates(series)
            if parsed.notna().any():
                columns.append(column)
    return columns


def _numeric_or_date_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            result[column] = pd.to_numeric(series, errors="coerce")
        elif pd.api.types.is_datetime64_any_dtype(series) or "date" in str(column).lower() or column in {
            "issue_d",
            "application_date",
        }:
            parsed = _parse_dates(series)
            result[column] = parsed.astype("int64").astype(float)
            result.loc[parsed.isna(), column] = np.nan
        else:
            result[column] = pd.to_numeric(series, errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def _parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="%Y-%m-%d")
    if parsed.notna().any():
        return parsed
    parsed = pd.to_datetime(series, errors="coerce", format="%Y-%m")
    if parsed.notna().any():
        return parsed
    return pd.to_datetime(series, errors="coerce", format="%b-%Y")


def _one_hot_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_parts = []
    test_parts = []
    for column in X_train.columns:
        train_values = X_train[column].astype("string").fillna("missing")
        test_values = X_test[column].astype("string").fillna("missing")
        categories = sorted(train_values.unique().tolist())
        train_encoded = np.zeros((len(X_train), len(categories)), dtype=float)
        test_encoded = np.zeros((len(X_test), len(categories)), dtype=float)
        category_to_index = {category: index for index, category in enumerate(categories)}

        for row_index, value in enumerate(train_values):
            train_encoded[row_index, category_to_index[value]] = 1.0
        for row_index, value in enumerate(test_values):
            index = category_to_index.get(value)
            if index is not None:
                test_encoded[row_index, index] = 1.0

        train_parts.append(train_encoded)
        test_parts.append(test_encoded)

    return np.hstack(train_parts), np.hstack(test_parts)
