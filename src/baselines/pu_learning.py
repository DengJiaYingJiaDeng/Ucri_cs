from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone


def elkan_noto_correction(
    model,
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
):
    """Elkan-Noto correction for rejected-applicant PU-style baselines."""
    x_labeled, y_labeled, x_unlabeled = _validate_labeled_unlabeled(X_labeled, y_labeled, X_unlabeled)
    positive_mask = y_labeled == 1
    if not positive_mask.any():
        raise ValueError("y_labeled must contain at least one positive label.")

    model.fit(x_labeled, y_labeled)
    unlabeled_probs = _predict_positive_probability(model, x_unlabeled)
    positive_probs = _predict_positive_probability(model, x_labeled.loc[positive_mask])
    c = max(float(positive_probs.mean()), 1e-6)
    corrected_probs = np.clip(unlabeled_probs / c, 0.0, 1.0)
    return model, corrected_probs


def pu_bagging(
    base_model,
    X_positive: pd.DataFrame,
    X_unlabeled: pd.DataFrame,
    n_bags: int = 50,
    bag_size_ratio: float = 0.5,
    random_state: int | None = 42,
):
    """Train PU bagging models by treating sampled unlabeled rows as weak negatives."""
    x_positive = pd.DataFrame(X_positive).copy().reset_index(drop=True)
    x_unlabeled = pd.DataFrame(X_unlabeled).copy().reset_index(drop=True).reindex(columns=x_positive.columns)
    if len(x_positive) == 0:
        raise ValueError("X_positive must not be empty.")
    if len(x_unlabeled) == 0:
        raise ValueError("X_unlabeled must not be empty.")
    n_bags = _validate_positive_integer("n_bags", n_bags)
    bag_size_ratio = _validate_probability_open("bag_size_ratio", bag_size_ratio)

    rng = np.random.default_rng(random_state)
    n_samples = max(1, int(round(len(x_unlabeled) * bag_size_ratio)))
    n_samples = min(n_samples, len(x_unlabeled))

    models = []
    for _ in range(n_bags):
        negative_indices = rng.choice(len(x_unlabeled), size=n_samples, replace=False)
        x_negative = x_unlabeled.iloc[negative_indices]
        x_train = pd.concat([x_positive, x_negative], ignore_index=True)
        y_train = np.concatenate([np.ones(len(x_positive), dtype=int), np.zeros(len(x_negative), dtype=int)])

        model = clone(base_model)
        model.fit(x_train, y_train)
        models.append(model)

    return models


def pu_bagging_predict(models: list, X: pd.DataFrame) -> np.ndarray:
    """Average positive-class probabilities from a PU bagging ensemble."""
    if len(models) == 0:
        raise ValueError("models must not be empty.")
    x = pd.DataFrame(X).copy()
    if len(x) == 0:
        raise ValueError("X must not be empty.")

    predictions = np.column_stack([_predict_positive_probability(model, x) for model in models])
    return predictions.mean(axis=1)


def upu_loss(y_pred: np.ndarray, y_true: np.ndarray, pi_p: float) -> float:
    """Unbiased PU risk estimator for positive-vs-unlabeled diagnostics."""
    y_pred, y_true = _validate_pu_loss_inputs(y_pred, y_true, pi_p)
    positive = y_true == 1
    unlabeled = y_true == 0

    risk_p = -np.mean(np.log(y_pred[positive] + 1e-10))
    risk_u = -np.mean(np.log(1.0 - y_pred[unlabeled] + 1e-10))
    risk_n = -pi_p * np.mean(np.log(y_pred[unlabeled] + 1e-10))
    return float(risk_p + risk_u - risk_n)


def nnpu_loss(y_pred: np.ndarray, y_true: np.ndarray, pi_p: float) -> float:
    """Non-negative PU risk estimator."""
    return float(max(0.0, upu_loss(y_pred, y_true, pi_p)))


def _validate_labeled_unlabeled(
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
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_labeled must contain binary 0/1 labels.")
    return labels.astype(int)


def _validate_pu_loss_inputs(y_pred: np.ndarray, y_true: np.ndarray, pi_p: float) -> tuple[np.ndarray, np.ndarray]:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true)
    if y_pred.ndim != 1 or y_true.ndim != 1:
        raise ValueError("y_pred and y_true must be one-dimensional arrays.")
    if len(y_pred) != len(y_true):
        raise ValueError("y_pred and y_true must have the same length.")
    if len(y_pred) == 0:
        raise ValueError("y_pred and y_true must not be empty.")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred must contain finite values.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain binary 0/1 values.")
    if not (y_true == 1).any() or not (y_true == 0).any():
        raise ValueError("y_true must contain both positive and unlabeled examples.")

    pi_p = float(pi_p)
    if not np.isfinite(pi_p) or pi_p <= 0 or pi_p >= 1:
        raise ValueError("pi_p must be in (0, 1).")
    return np.clip(y_pred, 1e-10, 1.0 - 1e-10), y_true.astype(int)


def _validate_positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _validate_probability_open(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0 or value > 1:
        raise ValueError(f"{name} must be in (0, 1].")
    return value


def _predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.ndim == 2:
        if probabilities.shape[1] < 2:
            raise ValueError("predict_proba must return a positive-class probability column.")
        probabilities = probabilities[:, 1]
    if probabilities.ndim != 1:
        raise ValueError("predicted probabilities must be a one-dimensional array.")
    if len(probabilities) != len(X):
        raise ValueError("predicted probabilities must have the same length as X.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("predicted probabilities must contain finite values.")
    return np.clip(probabilities, 0.0, 1.0)


PU_BASELINES = {
    "elkan_noto": elkan_noto_correction,
    "pu_bagging": pu_bagging,
    "upu_loss": upu_loss,
    "nnpu_loss": nnpu_loss,
}
