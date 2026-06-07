from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.baselines.traditional import TRADITIONAL_BASELINES
from src.evaluation.metrics import compute_all_metrics


@dataclass
class ProtocolResult:
    protocol: str
    model_name: str
    metrics: dict[str, float]
    predictions: np.ndarray
    true_labels: np.ndarray


def run_protocol_1(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_names: list[str] | None = None,
    random_state: int = 42,
) -> list[ProtocolResult]:
    """Run Protocol 1: accepted-only out-of-time PD benchmark."""
    x_train, y_train = _validate_feature_label_pair("train", X_train, y_train)
    _validate_feature_label_pair("validation", X_val, y_val)
    x_test, y_test = _validate_feature_label_pair("test", X_test, y_test)

    x_test = x_test.reindex(columns=x_train.columns)
    selected_models = model_names or ["LogisticRegression", "LightGBM", "CatBoost"]
    _validate_model_names(selected_models)

    results = []
    for name in selected_models:
        model = TRADITIONAL_BASELINES[name](random_state=random_state)
        model.fit(x_train, y_train)
        predictions = _predict_positive_probability(model, x_test)
        metrics = compute_all_metrics(y_test, predictions)
        results.append(
            ProtocolResult(
                protocol="Protocol1",
                model_name=name,
                metrics=metrics,
                predictions=predictions,
                true_labels=y_test.copy(),
            )
        )
    return results


def _validate_feature_label_pair(name: str, X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    x = pd.DataFrame(X).copy().reset_index(drop=True)
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError(f"{name} labels must be a one-dimensional array.")
    if len(x) != len(labels):
        raise ValueError(f"{name} features and labels must have the same length.")
    if len(x) == 0:
        raise ValueError(f"{name} features and labels must not be empty.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{name} labels must be binary 0/1 values.")
    if not x.columns.is_unique:
        raise ValueError(f"{name} features must have unique column names.")
    return x, labels.astype(int)


def _validate_model_names(model_names: list[str]) -> None:
    if len(model_names) == 0:
        raise ValueError("model_names must not be empty.")
    missing = [name for name in model_names if name not in TRADITIONAL_BASELINES]
    if missing:
        raise KeyError(f"Unknown model name(s): {missing}")


def _predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.ndim == 2:
        if probabilities.shape[1] < 2:
            raise ValueError("predict_proba must return a positive-class probability column.")
        probabilities = probabilities[:, 1]
    if probabilities.ndim != 1:
        raise ValueError("predictions must be a one-dimensional array.")
    if len(probabilities) != len(X):
        raise ValueError("predictions must have the same length as X.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("predictions must contain finite values.")
    return np.clip(probabilities, 0.0, 1.0)
