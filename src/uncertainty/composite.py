from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import minimize
from scipy.stats import rankdata


class CompositeUncertainty:
    """Quantile-normalized four-component uncertainty score."""

    COMPONENT_KEYS = ("variance", "entropy", "margin", "distance")

    def __init__(self, alpha: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)):
        self.alpha = self._validate_alpha(alpha)
        self._ref_train_distances: np.ndarray | None = None
        self._ref_train_signature: tuple[tuple[str, ...], int] | None = None

    def _validate_alpha(self, alpha: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        alpha_array = np.asarray(alpha, dtype=float)
        if alpha_array.shape != (4,):
            raise ValueError("Alpha must contain four weights.")
        if not np.all(np.isfinite(alpha_array)):
            raise ValueError("Alpha must contain finite weights.")
        if np.any(alpha_array < 0):
            raise ValueError("Alpha weights must be non-negative.")
        if not np.isclose(alpha_array.sum(), 1.0, atol=1e-6):
            raise ValueError(f"Alpha must sum to 1, got {alpha}.")
        return tuple(float(value) for value in alpha_array)

    def _as_1d_array(self, name: str, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array.")
        if len(array) == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite values.")
        return array

    def _validate_components(self, components: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        validated = {}
        lengths = set()
        for key in self.COMPONENT_KEYS:
            if key not in components:
                raise KeyError(key)
            values = self._as_1d_array(key, components[key])
            validated[key] = values
            lengths.add(len(values))
        if len(lengths) != 1:
            raise ValueError("Uncertainty components must have the same length.")
        return validated

    def _quantile_normalize(self, values: np.ndarray) -> np.ndarray:
        """Map a component to [0, 1] by empirical rank within the current batch."""
        array = self._as_1d_array("values", values)
        if len(array) == 1 or np.allclose(array, array[0]):
            return np.zeros_like(array, dtype=float)

        ranks = rankdata(array, method="average")
        normalized = (ranks - 1.0) / (len(array) - 1.0)
        return np.clip(normalized, 0.0, 1.0)

    def compute(self, components: Mapping[str, np.ndarray]) -> np.ndarray:
        validated = self._validate_components(components)
        normalized = {key: self._quantile_normalize(validated[key]) for key in self.COMPONENT_KEYS}
        return self._combine_normalized(normalized)

    def compute_from_teacher(
        self,
        X_test: pd.DataFrame,
        X_train: pd.DataFrame,
        teacher,
    ) -> np.ndarray:
        from src.uncertainty.distance import (
            compute_knn_distance_uncertainty,
            normalize_distance_against_reference,
        )

        teacher_uncertainty = teacher.compute_uncertainty(X_test)
        distance_uncertainty = compute_knn_distance_uncertainty(X_train, X_test, k=10)

        self._ensure_reference_distances(X_train)
        assert self._ref_train_distances is not None
        distance_normalized = normalize_distance_against_reference(
            self._ref_train_distances,
            distance_uncertainty,
        )

        components = {
            "variance": self._quantile_normalize(teacher_uncertainty["variance"]),
            "entropy": self._quantile_normalize(teacher_uncertainty["entropy"]),
            "margin": self._quantile_normalize(teacher_uncertainty["margin"]),
            "distance": self._as_1d_array("distance", distance_normalized),
        }
        return self._combine_normalized(components)

    def fit_alpha(
        self,
        X_sim_rej: pd.DataFrame,
        X_train: pd.DataFrame,
        teacher,
        y_hidden: np.ndarray,
    ) -> CompositeUncertainty:
        """Learn alpha by maximizing precision over low-uncertainty coverage slices."""
        y_hidden = self._as_1d_array("y_hidden", y_hidden).astype(int)
        if len(X_sim_rej) != len(y_hidden):
            raise ValueError("X_sim_rej and y_hidden must have the same length.")

        from src.uncertainty.distance import (
            compute_knn_distance_uncertainty,
            normalize_distance_against_reference,
        )

        teacher_uncertainty = teacher.compute_uncertainty(X_sim_rej)
        distance_uncertainty = compute_knn_distance_uncertainty(X_train, X_sim_rej, k=10)
        self._ensure_reference_distances(X_train)
        assert self._ref_train_distances is not None
        distance_normalized = normalize_distance_against_reference(
            self._ref_train_distances,
            distance_uncertainty,
        )

        base_components = {
            "variance": self._quantile_normalize(teacher_uncertainty["variance"]),
            "entropy": self._quantile_normalize(teacher_uncertainty["entropy"]),
            "margin": self._quantile_normalize(teacher_uncertainty["margin"]),
            "distance": self._as_1d_array("distance", distance_normalized),
        }
        base_components = self._validate_components(base_components)

        teacher_probs = self._as_1d_array("teacher_probs", teacher.predict_proba(X_sim_rej))
        if len(teacher_probs) != len(y_hidden):
            raise ValueError("teacher probabilities and y_hidden must have the same length.")
        pseudo_labels = (teacher_probs >= 0.5).astype(int)
        correct = (pseudo_labels == y_hidden).astype(float)

        coverages = np.array([0.2, 0.3, 0.4, 0.5])

        def objective(alpha_vec: np.ndarray) -> float:
            alpha = self._project_to_simplex(alpha_vec)
            uncertainty = self._combine_normalized(base_components, alpha=alpha)
            precisions = [self._precision_at_coverage(uncertainty, correct, coverage) for coverage in coverages]
            return -float(trapezoid(precisions, coverages) / (coverages[-1] - coverages[0]))

        result = minimize(
            objective,
            x0=np.asarray(self.alpha, dtype=float),
            method="Nelder-Mead",
            options={"maxiter": 200, "xatol": 0.01},
        )
        self.alpha = tuple(float(value) for value in self._project_to_simplex(result.x))
        return self

    def _combine_normalized(
        self,
        normalized_components: Mapping[str, np.ndarray],
        alpha: np.ndarray | tuple[float, float, float, float] | None = None,
    ) -> np.ndarray:
        validated = self._validate_components(normalized_components)
        alpha_array = np.asarray(self.alpha if alpha is None else alpha, dtype=float)
        combined = np.zeros(len(validated[self.COMPONENT_KEYS[0]]), dtype=float)
        for index, key in enumerate(self.COMPONENT_KEYS):
            combined += alpha_array[index] * validated[key]
        return np.clip(combined, 0.0, 1.0)

    def _ensure_reference_distances(self, X_train: pd.DataFrame) -> None:
        signature = self._training_signature(X_train)
        if self._ref_train_distances is not None and self._ref_train_signature == signature:
            return

        from src.uncertainty.distance import compute_knn_distance_uncertainty

        self._ref_train_distances = compute_knn_distance_uncertainty(X_train, X_train, k=10)
        self._ref_train_signature = signature

    def _training_signature(self, X_train: pd.DataFrame) -> tuple[tuple[str, ...], int]:
        frame = pd.DataFrame(X_train)
        return tuple(frame.columns), len(frame)

    def _project_to_simplex(self, values: np.ndarray) -> np.ndarray:
        weights = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.maximum(weights, 0.0)
        total = weights.sum()
        if total <= 0:
            return np.full(4, 0.25)
        return weights / total

    def _precision_at_coverage(self, uncertainty: np.ndarray, correct: np.ndarray, coverage: float) -> float:
        if not 0 < coverage <= 1:
            raise ValueError("coverage must be in (0, 1].")

        n_selected = max(1, int(np.ceil(len(uncertainty) * coverage)))
        selected_indices = np.argpartition(uncertainty, n_selected - 1)[:n_selected]
        return float(correct[selected_indices].mean())
