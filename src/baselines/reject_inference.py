from __future__ import annotations

import numpy as np
import pandas as pd


def hard_augmentation(
    model,
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
    threshold: float = 0.5,
):
    """Classic hard reject-inference augmentation.

    The accepted-only model scores rejected applicants, assigns hard pseudo-labels,
    then refits on accepted labels plus pseudo-labeled rejected rows.
    """
    x_labeled, y_labeled, x_unlabeled = _validate_training_inputs(X_labeled, y_labeled, X_unlabeled)
    threshold = _validate_probability_threshold("threshold", threshold)

    model.fit(x_labeled, y_labeled)
    probabilities = _predict_positive_probability(model, x_unlabeled)
    hard_labels = (probabilities >= threshold).astype(int)

    x_all = pd.concat([x_labeled, x_unlabeled], ignore_index=True)
    y_all = np.concatenate([y_labeled, hard_labels])
    model.fit(x_all, y_all)
    return model


def fuzzy_augmentation(
    model,
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
):
    """Fuzzy augmentation using model-estimated rejected bad-rate weights."""
    x_labeled, y_labeled, x_unlabeled = _validate_training_inputs(X_labeled, y_labeled, X_unlabeled)

    model.fit(x_labeled, y_labeled)
    probabilities = _predict_positive_probability(model, x_unlabeled)

    x_all, y_all, sample_weight = _expand_unlabeled_soft_labels(
        x_labeled,
        y_labeled,
        x_unlabeled,
        probabilities,
    )
    _fit_with_optional_sample_weight(model, x_all, y_all, sample_weight)
    return model


def parceling(
    model,
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
    n_bins: int = 10,
):
    """Risk-score parceling baseline with bin-level rejected bad-rate labels."""
    x_labeled, y_labeled, x_unlabeled = _validate_training_inputs(X_labeled, y_labeled, X_unlabeled)
    n_bins = _validate_n_bins(n_bins)

    model.fit(x_labeled, y_labeled)
    probabilities = _predict_positive_probability(model, x_unlabeled)
    parcel_probabilities = _compute_parcel_probabilities(probabilities, n_bins)

    x_all, y_all, sample_weight = _expand_unlabeled_soft_labels(
        x_labeled,
        y_labeled,
        x_unlabeled,
        parcel_probabilities,
    )
    _fit_with_optional_sample_weight(model, x_all, y_all, sample_weight)
    return model


def self_training(
    model,
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
    n_iterations: int = 3,
    confidence_threshold: float = 0.8,
):
    """Vanilla self-training baseline without uncertainty-aware filtering."""
    x_labeled, y_labeled, current_unlabeled = _validate_training_inputs(X_labeled, y_labeled, X_unlabeled)
    n_iterations = _validate_n_iterations(n_iterations)
    confidence_threshold = _validate_confidence_threshold(confidence_threshold)

    model.fit(x_labeled, y_labeled)
    for _ in range(n_iterations):
        if len(current_unlabeled) == 0:
            break

        probabilities = _predict_positive_probability(model, current_unlabeled)
        confident = (probabilities >= confidence_threshold) | (probabilities <= 1.0 - confidence_threshold)
        if not confident.any():
            break

        hard_labels = (probabilities[confident] >= 0.5).astype(int)
        x_labeled = pd.concat([x_labeled, current_unlabeled.iloc[np.flatnonzero(confident)]], ignore_index=True)
        y_labeled = np.concatenate([y_labeled, hard_labels])
        current_unlabeled = current_unlabeled.iloc[np.flatnonzero(~confident)].reset_index(drop=True)
        model.fit(x_labeled, y_labeled)

    return model


def ipw_weighted_pd(
    propensity_model,
    pd_model,
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    eps: float = 0.01,
):
    """Accepted-only PD baseline reweighted by inverse approval propensity."""
    x_labeled = pd.DataFrame(X_labeled).copy().reset_index(drop=True)
    y_labeled = _validate_binary_labels(y_labeled, expected_length=len(x_labeled))
    eps = _validate_positive_eps(eps)

    propensity = _predict_propensity(propensity_model, x_labeled)
    weights = 1.0 / np.maximum(propensity, eps)
    _fit_with_optional_sample_weight(pd_model, x_labeled, y_labeled, weights)
    return pd_model


def _validate_training_inputs(
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    x_labeled = pd.DataFrame(X_labeled).copy().reset_index(drop=True)
    x_unlabeled = pd.DataFrame(X_unlabeled).copy().reset_index(drop=True)
    y_labeled = _validate_binary_labels(y_labeled, expected_length=len(x_labeled))

    if len(x_labeled) == 0:
        raise ValueError("X_labeled must not be empty.")
    if len(x_unlabeled) == 0:
        raise ValueError("X_unlabeled must not be empty.")

    x_unlabeled = x_unlabeled.reindex(columns=x_labeled.columns)
    return x_labeled, y_labeled, x_unlabeled


def _validate_binary_labels(labels: np.ndarray, expected_length: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("y_labeled must be a one-dimensional array.")
    if len(labels) != expected_length:
        raise ValueError("X_labeled and y_labeled must have the same length.")
    if len(labels) == 0:
        raise ValueError("y_labeled must not be empty.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_labeled must contain binary 0/1 labels.")
    return labels.astype(int)


def _validate_probability_threshold(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"{name} must be in [0, 1].")
    return value


def _validate_confidence_threshold(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.5 or value > 1.0:
        raise ValueError("confidence_threshold must be in [0.5, 1.0].")
    return value


def _validate_n_bins(n_bins: int) -> int:
    if isinstance(n_bins, bool):
        raise ValueError("n_bins must be a positive integer.")
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    return n_bins


def _validate_n_iterations(n_iterations: int) -> int:
    if isinstance(n_iterations, bool):
        raise ValueError("n_iterations must be a non-negative integer.")
    n_iterations = int(n_iterations)
    if n_iterations < 0:
        raise ValueError("n_iterations must be a non-negative integer.")
    return n_iterations


def _validate_positive_eps(eps: float) -> float:
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive.")
    return eps


def _predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.ndim == 2:
        if probabilities.shape[1] < 2:
            raise ValueError("predict_proba must return a positive-class probability column.")
        probabilities = probabilities[:, 1]
    probabilities = _validate_probability_vector(probabilities, expected_length=len(X), name="predicted probabilities")
    return np.clip(probabilities, 0.0, 1.0)


def _predict_propensity(propensity_model, X: pd.DataFrame) -> np.ndarray:
    propensity = np.asarray(propensity_model.predict_proba(X), dtype=float)
    if propensity.ndim == 2:
        if propensity.shape[1] < 2:
            raise ValueError("propensity predict_proba must return a positive-class probability column.")
        propensity = propensity[:, 1]
    propensity = _validate_probability_vector(propensity, expected_length=len(X), name="propensity")
    return np.clip(propensity, 0.0, 1.0)


def _validate_probability_vector(values: np.ndarray, expected_length: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if len(values) != expected_length:
        raise ValueError(f"{name} must have the same length as X.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values.")
    return values


def _compute_parcel_probabilities(probabilities: np.ndarray, n_bins: int) -> np.ndarray:
    n_bins = min(n_bins, len(probabilities))
    ranks = pd.Series(probabilities).rank(method="first").to_numpy() - 1
    bin_ids = np.floor(ranks * n_bins / len(probabilities)).astype(int)
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    parcel_probabilities = np.zeros(len(probabilities), dtype=float)
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.any():
            parcel_probabilities[mask] = probabilities[mask].mean()
    return np.clip(parcel_probabilities, 0.0, 1.0)


def _expand_unlabeled_soft_labels(
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
    probabilities: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    probabilities = _validate_probability_vector(probabilities, expected_length=len(X_unlabeled), name="soft labels")
    probabilities = np.clip(probabilities, 0.0, 1.0)

    x_all = pd.concat([X_labeled, X_unlabeled, X_unlabeled], ignore_index=True)
    y_all = np.concatenate(
        [
            y_labeled,
            np.zeros(len(X_unlabeled), dtype=int),
            np.ones(len(X_unlabeled), dtype=int),
        ]
    )
    sample_weight = np.concatenate(
        [
            np.ones(len(X_labeled), dtype=float),
            1.0 - probabilities,
            probabilities,
        ]
    )
    return x_all, y_all, sample_weight


def _fit_with_optional_sample_weight(model, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray) -> None:
    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)


REJECT_INFERENCE_BASELINES = {
    "hard": hard_augmentation,
    "fuzzy": fuzzy_augmentation,
    "parceling": parceling,
    "self_training": self_training,
    "ipw": ipw_weighted_pd,
}
