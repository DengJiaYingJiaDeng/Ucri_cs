from __future__ import annotations

import numpy as np
import pandas as pd


class PseudoLabeler:
    """Uncertainty-aware soft pseudo-labeling for rejected applicants."""

    def __init__(
        self,
        tau_u: float = 0.5,
        gamma: float = 2.0,
        theta_low: float | None = None,
        theta_high: float | None = None,
    ):
        self.tau_u = self._validate_probability_threshold("tau_u", tau_u)
        self.gamma = self._validate_gamma(gamma)
        self.theta_low = None if theta_low is None else self._validate_probability_threshold("theta_low", theta_low)
        self.theta_high = None if theta_high is None else self._validate_probability_threshold("theta_high", theta_high)
        if (self.theta_low is None) != (self.theta_high is None):
            raise ValueError("theta_low and theta_high must be provided together.")
        if self.theta_low is not None and self.theta_low > self.theta_high:
            raise ValueError("theta_low must be less than or equal to theta_high.")

    def label(
        self,
        X: pd.DataFrame,
        teacher_probs: np.ndarray,
        uncertainty: np.ndarray,
    ) -> dict[str, np.ndarray]:
        n_samples = len(pd.DataFrame(X))
        probabilities, uncertainty = self._validate_label_inputs(n_samples, teacher_probs, uncertainty)

        selected = uncertainty < self.tau_u
        soft_labels = np.clip(probabilities, 1e-6, 1 - 1e-6)
        weights = np.zeros(n_samples, dtype=float)
        weights[selected] = np.exp(-self.gamma * uncertainty[selected])

        return {
            "soft_label": soft_labels,
            "weight": weights,
            "decision": self._make_decisions(soft_labels, uncertainty, selected),
        }

    def compute_coverage(self, weights: np.ndarray) -> float:
        weights = self._as_1d_array("weights", weights)
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        return float((weights > 0).mean())

    def tau_sensitivity(
        self,
        X: pd.DataFrame,
        teacher_probs: np.ndarray,
        uncertainty: np.ndarray,
        true_labels: np.ndarray,
        tau_values: list[float] | None = None,
    ) -> dict[str, list[dict[str, float | None]]]:
        """Scan uncertainty thresholds and report pseudo-label quality diagnostics."""
        tau_values = tau_values or [0.1, 0.2, 0.3, 0.4, 0.5]
        true_labels = self._validate_true_labels(true_labels, expected_length=len(pd.DataFrame(X)))
        saved_tau = self.tau_u
        results = []

        try:
            for tau in tau_values:
                self.tau_u = self._validate_probability_threshold("tau_u", tau)
                result = self.label(X, teacher_probs, uncertainty)
                coverage = self.compute_coverage(result["weight"])
                entry: dict[str, float | None] = {
                    "tau_u": float(tau),
                    "coverage": coverage,
                    "precision": None,
                    "ece": None,
                }
                if coverage > 0:
                    mask = result["weight"] > 0
                    predicted_labels = (result["soft_label"] >= 0.5).astype(int)
                    entry["precision"] = float((predicted_labels[mask] == true_labels[mask]).mean())
                    from src.evaluation.metrics import compute_ece

                    entry["ece"] = compute_ece(true_labels[mask], result["soft_label"][mask])
                results.append(entry)
        finally:
            self.tau_u = saved_tau

        return {"tau_sensitivity": results}

    def coverage_constrained_label(
        self,
        X: pd.DataFrame,
        teacher_probs: np.ndarray,
        uncertainty: np.ndarray,
        coverage_target: float = 0.3,
    ) -> dict[str, np.ndarray | float]:
        """Select the lowest-uncertainty fraction when a fixed threshold is too strict."""
        coverage_target = self._validate_coverage_target(coverage_target)
        n_samples = len(pd.DataFrame(X))
        probabilities, uncertainty = self._validate_label_inputs(n_samples, teacher_probs, uncertainty)

        n_selected = max(1, int(n_samples * coverage_target))
        selected_indices = np.argsort(uncertainty, kind="mergesort")[:n_selected]
        selected = np.zeros(n_samples, dtype=bool)
        selected[selected_indices] = True

        soft_labels = np.clip(probabilities, 1e-6, 1 - 1e-6)
        weights = np.zeros(n_samples, dtype=float)
        weights[selected] = np.exp(-self.gamma * uncertainty[selected])

        return {
            "soft_label": soft_labels,
            "weight": weights,
            "decision": self._make_decisions(soft_labels, uncertainty, selected),
            "coverage": float(selected.mean()),
        }

    def precision_coverage_curve(
        self,
        X: pd.DataFrame,
        teacher_probs: np.ndarray,
        uncertainty: np.ndarray,
        true_labels: np.ndarray,
        tau_values: list[float] | np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Return precision, coverage, and calibration error across uncertainty thresholds."""
        tau_values = np.asarray(tau_values if tau_values is not None else np.linspace(0.05, 0.95, 20), dtype=float)
        true_labels = self._validate_true_labels(true_labels, expected_length=len(pd.DataFrame(X)))
        saved_tau = self.tau_u
        curve = {"tau_u": [], "coverage": [], "precision": [], "ece": []}

        try:
            for tau in tau_values:
                self.tau_u = self._validate_probability_threshold("tau_u", float(tau))
                result = self.label(X, teacher_probs, uncertainty)
                mask = result["weight"] > 0
                coverage = self.compute_coverage(result["weight"])

                if mask.any():
                    predicted_labels = (result["soft_label"] >= 0.5).astype(int)
                    precision = float((predicted_labels[mask] == true_labels[mask]).mean())
                    from src.evaluation.metrics import compute_ece

                    ece = compute_ece(true_labels[mask], result["soft_label"][mask])
                else:
                    precision = float("nan")
                    ece = float("nan")

                curve["tau_u"].append(float(tau))
                curve["coverage"].append(coverage)
                curve["precision"].append(precision)
                curve["ece"].append(ece)
        finally:
            self.tau_u = saved_tau

        return curve

    def _make_decisions(
        self,
        soft_labels: np.ndarray,
        uncertainty: np.ndarray,
        selected: np.ndarray,
    ) -> np.ndarray:
        decisions = np.full(len(soft_labels), "manual_review", dtype=object)
        if self.theta_low is None or self.theta_high is None:
            return decisions

        low_pd = selected & (uncertainty < self.tau_u) & (soft_labels <= self.theta_low)
        high_pd = selected & (uncertainty < self.tau_u) & (soft_labels >= self.theta_high)
        decisions[low_pd] = "approve"
        decisions[high_pd] = "reject"
        return decisions

    def _validate_label_inputs(
        self,
        n_samples: int,
        teacher_probs: np.ndarray,
        uncertainty: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self._as_1d_array("teacher_probs", teacher_probs)
        uncertainty = self._as_1d_array("uncertainty", uncertainty)
        if len(probabilities) != n_samples or len(uncertainty) != n_samples:
            raise ValueError("X, teacher_probs, and uncertainty must have the same length.")
        if np.any((uncertainty < 0) | (uncertainty > 1)):
            raise ValueError("uncertainty must be normalized to [0, 1].")
        return probabilities, uncertainty

    def _validate_true_labels(self, true_labels: np.ndarray, expected_length: int) -> np.ndarray:
        labels = self._as_1d_array("true_labels", true_labels).astype(int)
        if len(labels) != expected_length:
            raise ValueError("X and true_labels must have the same length.")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("true_labels must be binary 0/1 values.")
        return labels

    def _as_1d_array(self, name: str, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array.")
        if len(array) == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite values.")
        return array

    def _validate_probability_threshold(self, name: str, value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or value < 0 or value > 1:
            raise ValueError(f"{name} must be in [0, 1].")
        return value

    def _validate_gamma(self, gamma: float) -> float:
        gamma = float(gamma)
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be a non-negative finite value.")
        return gamma

    def _validate_coverage_target(self, coverage_target: float) -> float:
        coverage_target = float(coverage_target)
        if not np.isfinite(coverage_target) or coverage_target <= 0 or coverage_target > 1:
            raise ValueError("coverage_target must be in (0, 1].")
        return coverage_target
