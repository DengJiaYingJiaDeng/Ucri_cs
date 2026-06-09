import numpy as np
import pytest

from src.decision.threshold import DecisionThresholdOptimizer, DecisionThresholds


def test_optimizer_finds_thresholds():
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = np.linspace(0.01, 0.5, 100)
    optimizer = DecisionThresholdOptimizer(target_bad_rate=0.1)

    thresholds = optimizer.optimize(y_true, y_pred)

    assert "theta_reject" in thresholds
    assert "theta_approve" in thresholds
    assert 0 <= thresholds.theta_approve <= thresholds.theta_reject <= 1


def test_optimizer_constrains_bad_rate():
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = np.append(np.linspace(0.01, 0.3, 80), np.linspace(0.3, 0.99, 20))
    optimizer = DecisionThresholdOptimizer(target_bad_rate=0.15, min_approval_rate=0.3)

    thresholds = optimizer.optimize(y_true, y_pred)
    decisions = optimizer.apply(y_pred, thresholds)

    approved = decisions == "approve"
    assert approved.mean() >= 0.3
    assert y_true[approved].mean() <= 0.15


def test_optimizer_outputs_manual_review():
    y_true = np.array([0] * 50 + [1] * 50)
    y_pred = np.linspace(0.01, 0.99, 100)
    optimizer = DecisionThresholdOptimizer(target_bad_rate=0.2, tau_u=0.5, tau_decision_multiplier=1.0)

    thresholds = optimizer.optimize(y_true, y_pred)
    decisions = optimizer.apply(y_pred, thresholds)

    assert set(decisions) <= {"approve", "reject", "manual_review"}
    assert "manual_review" in decisions


def test_optimizer_matches_bruteforce_threshold_search_with_unique_scores():
    y_true = np.array([0, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([0.03, 0.11, 0.15, 0.22, 0.31, 0.33, 0.62, 0.80])
    optimizer = DecisionThresholdOptimizer(target_bad_rate=0.25, min_approval_rate=0.25)

    thresholds = optimizer.optimize(y_true, y_pred)
    expected_theta = _bruteforce_theta_approve(y_true, y_pred, target_bad_rate=0.25, min_approval_rate=0.25)

    assert thresholds.theta_approve == pytest.approx(expected_theta)


def test_optimizer_prefers_least_bad_rate_violation_when_constraints_are_infeasible():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0.05, 0.10, 0.20, 0.40, 0.90])
    optimizer = DecisionThresholdOptimizer(target_bad_rate=0.10, min_approval_rate=0.80)

    thresholds = optimizer.optimize(y_true, y_pred)
    decisions = optimizer.apply(y_pred, thresholds)

    approved = decisions == "approve"
    assert thresholds.theta_approve == pytest.approx(0.40)
    assert approved.mean() == pytest.approx(0.80)
    assert y_true[approved].mean() == pytest.approx(0.50)


def test_optimizer_handles_large_validation_arrays_without_expanding_nested_scans():
    rng = np.random.default_rng(42)
    y_pred = rng.uniform(0.01, 0.99, 50_000)
    y_true = rng.binomial(1, y_pred)
    optimizer = DecisionThresholdOptimizer(target_bad_rate=0.12, min_approval_rate=0.30)

    thresholds = optimizer.optimize(y_true, y_pred)

    assert 0 <= thresholds.theta_approve <= thresholds.theta_reject <= 1


def test_uncertainty_blocks_approval_and_routes_high_uncertainty_to_review():
    y_pred = np.array([0.05, 0.06, 0.85, 0.30])
    uncertainty = np.array([0.1, 0.9, 0.1, 0.95])
    thresholds = DecisionThresholds(theta_approve=0.1, theta_reject=0.7)
    optimizer = DecisionThresholdOptimizer(tau_u=0.5, tau_decision_multiplier=1.0)

    decisions = optimizer.apply(y_pred, thresholds, uncertainty=uncertainty)

    assert decisions.tolist() == ["approve", "manual_review", "reject", "manual_review"]


def test_decision_sensitivity_outputs_multipliers():
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 50 + [1] * 50)
    y_pred = np.linspace(0.01, 0.99, 100)
    uncertainty = rng.uniform(0, 1, 100)
    optimizer = DecisionThresholdOptimizer(target_bad_rate=0.2, tau_u=0.5)

    results = optimizer.decision_sensitivity(y_true, y_pred, uncertainty)

    multipliers = [result["tau_decision_multiplier"] for result in results]
    assert multipliers == [1.0, 1.25, 1.5]
    assert all(result["tau_decision"] == pytest.approx(result["tau_decision_multiplier"] * 0.5) for result in results)
    assert all(
        result["approval_rate"] + result["reject_rate"] + result["manual_review_rate"] == pytest.approx(1.0)
        for result in results
    )


def test_optimizer_rejects_invalid_inputs():
    optimizer = DecisionThresholdOptimizer()

    with pytest.raises(ValueError, match="same length"):
        optimizer.optimize(np.array([0, 1]), np.array([0.1]))

    with pytest.raises(ValueError, match="target_bad_rate"):
        DecisionThresholdOptimizer(target_bad_rate=1.5)


def _bruteforce_theta_approve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_bad_rate: float,
    min_approval_rate: float,
) -> float:
    candidates = np.unique(y_pred)
    best_theta = float(np.quantile(y_pred, min(max(min_approval_rate, 0.0), 1.0)))
    best_key = None

    for theta in candidates:
        approved = y_pred <= theta
        approval_rate = float(approved.mean())
        if approved.sum() == 0 or approval_rate < min_approval_rate:
            continue

        realized_bad_rate = float(y_true[approved].mean())
        feasible = int(realized_bad_rate <= target_bad_rate)
        violation = max(0.0, realized_bad_rate - target_bad_rate)
        key = (feasible, approval_rate, -violation)
        if best_key is None or key > best_key:
            best_key = key
            best_theta = float(theta)

    if best_key is not None and best_key[0] == 1:
        return best_theta

    best_violation = float("inf")
    best_approval = -1.0
    for theta in candidates:
        approved = y_pred <= theta
        approval_rate = float(approved.mean())
        if approved.sum() == 0 or approval_rate < min_approval_rate:
            continue
        realized_bad_rate = float(y_true[approved].mean())
        violation = max(0.0, realized_bad_rate - target_bad_rate)
        if violation < best_violation or (np.isclose(violation, best_violation) and approval_rate > best_approval):
            best_violation = violation
            best_approval = approval_rate
            best_theta = float(theta)
    return best_theta
