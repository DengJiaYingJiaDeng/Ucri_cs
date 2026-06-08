from __future__ import annotations

import numpy as np
from scipy.special import expit


def negative_log_likelihood(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    """Binary NLL after temperature scaling."""
    logits = _as_1d_float_array("logits", logits)
    y = _as_binary_labels(y, expected_length=len(logits))
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be a positive finite value.")

    probabilities = apply_temperature(logits, temperature)
    probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
    return float(-np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities)))


def fit_temperature(
    logits: np.ndarray,
    y: np.ndarray,
    coarse_grid_size: int = 80,
    fine_grid_size: int = 80,
) -> float:
    """Fit a scalar temperature by grid-searching validation NLL."""
    logits = _as_1d_float_array("logits", logits)
    y = _as_binary_labels(y, expected_length=len(logits))
    if coarse_grid_size < 2 or fine_grid_size < 2:
        raise ValueError("coarse_grid_size and fine_grid_size must both be at least 2.")

    coarse_grid = np.geomspace(0.01, 10.0, coarse_grid_size)
    coarse_scores = np.array([negative_log_likelihood(logits, y, float(temperature)) for temperature in coarse_grid])
    best_temperature = float(coarse_grid[int(np.argmin(coarse_scores))])

    lower = max(0.01, best_temperature / 1.5)
    upper = min(10.0, best_temperature * 1.5)
    fine_grid = np.linspace(lower, upper, fine_grid_size)
    fine_scores = np.array([negative_log_likelihood(logits, y, float(temperature)) for temperature in fine_grid])
    return float(fine_grid[int(np.argmin(fine_scores))])


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Convert logits to probabilities with a positive scalar temperature."""
    logits = _as_1d_float_array("logits", logits)
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be a positive finite value.")
    return expit(logits / float(temperature))


def _as_1d_float_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _as_binary_labels(y: np.ndarray, expected_length: int) -> np.ndarray:
    labels = _as_1d_float_array("y", y)
    if len(labels) != expected_length:
        raise ValueError("logits and y must have the same length.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y must contain binary 0/1 labels.")
    return labels
