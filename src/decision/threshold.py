from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DecisionThresholds:
    theta_approve: float
    theta_reject: float

    def __contains__(self, key: str) -> bool:
        return key in {"theta_approve", "theta_reject"}

    def __getitem__(self, key: str) -> float:
        if key == "theta_approve":
            return self.theta_approve
        if key == "theta_reject":
            return self.theta_reject
        raise KeyError(key)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class DecisionThresholdOptimizer:
    """Optimize approve/reject thresholds under bad-rate and uncertainty controls."""

    def __init__(
        self,
        target_bad_rate: float = 0.08,
        min_approval_rate: float = 0.2,
        tau_u: float = 0.5,
        tau_decision_multiplier: float = 1.0,
    ):
        self.target_bad_rate = self._validate_probability("target_bad_rate", target_bad_rate)
        self.min_approval_rate = self._validate_probability("min_approval_rate", min_approval_rate)
        self.tau_u = self._validate_probability("tau_u", tau_u)
        self.tau_decision_multiplier = self._validate_positive("tau_decision_multiplier", tau_decision_multiplier)

    @property
    def tau_decision(self) -> float:
        return self.tau_u * self.tau_decision_multiplier

    def optimize(self, y_true: np.ndarray, y_pred: np.ndarray) -> DecisionThresholds:
        y_true, y_pred = self._validate_binary_inputs(y_true, y_pred)
        thresholds, approval_rates, bad_rates = self._threshold_summaries(y_true, y_pred)
        eligible = approval_rates >= self.min_approval_rate
        feasible = eligible & (bad_rates <= self.target_bad_rate)

        if feasible.any():
            best_theta = float(thresholds[np.flatnonzero(feasible)[-1]])
        elif eligible.any():
            best_theta = self._least_violating_threshold(
                thresholds[eligible],
                approval_rates[eligible],
                bad_rates[eligible],
            )
        else:
            best_theta = self._least_violating_threshold(thresholds, approval_rates, bad_rates)

        theta_reject = max(best_theta, float(np.quantile(y_pred, 0.75)), best_theta * 1.5)
        theta_reject = float(np.clip(theta_reject, best_theta, 1.0))
        return DecisionThresholds(theta_approve=float(np.clip(best_theta, 0.0, 1.0)), theta_reject=theta_reject)

    def apply(
        self,
        y_pred: np.ndarray,
        thresholds: DecisionThresholds | dict[str, float],
        uncertainty: np.ndarray | None = None,
    ) -> np.ndarray:
        y_pred = self._as_probability_array("y_pred", y_pred)
        thresholds = self._coerce_thresholds(thresholds)
        if thresholds.theta_approve > thresholds.theta_reject:
            raise ValueError("theta_approve must be less than or equal to theta_reject.")

        decisions = np.full(len(y_pred), "manual_review", dtype=object)
        approve_mask = y_pred <= thresholds.theta_approve
        reject_mask = y_pred >= thresholds.theta_reject

        if uncertainty is not None:
            uncertainty = self._as_probability_array("uncertainty", uncertainty)
            if len(uncertainty) != len(y_pred):
                raise ValueError("y_pred and uncertainty must have the same length.")
            high_uncertainty = uncertainty > self.tau_decision
            approve_mask = approve_mask & ~high_uncertainty
            reject_mask = reject_mask & ~high_uncertainty

        decisions[approve_mask] = "approve"
        decisions[reject_mask] = "reject"
        return decisions

    def decision_sensitivity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainty: np.ndarray,
        tau_multipliers: list[float] | None = None,
    ) -> list[dict[str, float]]:
        tau_multipliers = tau_multipliers or [1.0, 1.25, 1.5]
        saved_multiplier = self.tau_decision_multiplier
        results = []
        try:
            for multiplier in tau_multipliers:
                self.tau_decision_multiplier = self._validate_positive("tau_decision_multiplier", multiplier)
                thresholds = self.optimize(y_true, y_pred)
                decisions = self.apply(y_pred, thresholds, uncertainty)
                approved = decisions == "approve"
                results.append(
                    {
                        "tau_decision_multiplier": float(multiplier),
                        "tau_decision": float(self.tau_decision),
                        "theta_approve": thresholds.theta_approve,
                        "theta_reject": thresholds.theta_reject,
                        "approval_rate": float((decisions == "approve").mean()),
                        "reject_rate": float((decisions == "reject").mean()),
                        "manual_review_rate": float((decisions == "manual_review").mean()),
                        "approved_bad_rate": float(np.asarray(y_true)[approved].mean()) if approved.any() else float("nan"),
                    }
                )
        finally:
            self.tau_decision_multiplier = saved_multiplier
        return results

    def _candidate_thresholds(self, y_pred: np.ndarray) -> np.ndarray:
        percentiles = np.percentile(y_pred, np.linspace(1, 99, 99))
        candidates = np.unique(np.concatenate([y_pred, percentiles, np.array([0.0, 1.0])]))
        return np.sort(np.clip(candidates, 0.0, 1.0))

    def _threshold_summaries(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        order = np.argsort(y_pred, kind="mergesort")
        sorted_pred = y_pred[order]
        sorted_true = y_true[order]
        last_indices = np.flatnonzero(np.r_[sorted_pred[1:] != sorted_pred[:-1], True])

        thresholds = sorted_pred[last_indices]
        approved_counts = last_indices + 1
        bad_counts = np.cumsum(sorted_true)[last_indices]
        approval_rates = approved_counts / len(sorted_pred)
        bad_rates = bad_counts / approved_counts
        return thresholds, approval_rates, bad_rates

    def _least_violating_threshold(
        self,
        thresholds: np.ndarray,
        approval_rates: np.ndarray,
        bad_rates: np.ndarray,
    ) -> float:
        violations = np.maximum(0.0, bad_rates - self.target_bad_rate)
        best_violation = np.min(violations)
        tied = np.isclose(violations, best_violation)
        tied_indices = np.flatnonzero(tied)
        best_index = tied_indices[np.argmax(approval_rates[tied_indices])]
        return float(thresholds[best_index])

    def _coerce_thresholds(self, thresholds: DecisionThresholds | dict[str, float]) -> DecisionThresholds:
        if isinstance(thresholds, DecisionThresholds):
            return thresholds
        if not {"theta_approve", "theta_reject"}.issubset(thresholds):
            raise KeyError("thresholds must contain theta_approve and theta_reject.")
        return DecisionThresholds(
            theta_approve=self._validate_probability("theta_approve", thresholds["theta_approve"]),
            theta_reject=self._validate_probability("theta_reject", thresholds["theta_reject"]),
        )

    def _validate_binary_inputs(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_true = np.asarray(y_true)
        y_pred = self._as_probability_array("y_pred", y_pred)
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        if len(y_true) == 0:
            raise ValueError("inputs must not be empty.")
        if not np.isin(y_true, [0, 1]).all():
            raise ValueError("y_true must contain binary 0/1 labels.")
        return y_true.astype(int), y_pred

    def _as_probability_array(self, name: str, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array.")
        if len(array) == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite values.")
        if np.any((array < 0) | (array > 1)):
            raise ValueError(f"{name} must be normalized to [0, 1].")
        return array

    def _validate_probability(self, name: str, value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or value < 0 or value > 1:
            raise ValueError(f"{name} must be in [0, 1].")
        return value

    def _validate_positive(self, name: str, value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive.")
        return value
