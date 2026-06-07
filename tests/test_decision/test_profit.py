import numpy as np
import pytest

from src.decision.profit import (
    compute_expected_profit,
    compute_historical_profit,
    compute_oracle_profit,
    compute_oracle_profit_ratio,
    compute_profit_frontier,
    compute_random_profit,
)


def test_expected_profit_uses_approved_mask_and_cost_components():
    y_pred = np.array([0.05, 0.20, 0.70])
    y_true = np.array([0, 1, 0])
    approved = np.array([True, True, False])
    loan_amounts = np.array([10000.0, 20000.0, 30000.0])

    result = compute_expected_profit(
        y_pred,
        y_true,
        approved,
        loan_amounts,
        lgd=0.5,
        interest_rate=0.10,
        funding_cost=0.04,
        servicing_cost=0.01,
        prepayment_haircut=1.0,
        term_years=1.0,
    )

    expected_profit = (30000 * 0.10) - (30000 * 0.04) - (30000 * 0.01) - (20000 * 0.5)
    assert result["total_profit"] == pytest.approx(expected_profit)
    assert result["profit_per_loan"] == pytest.approx(expected_profit / 2)
    assert result["approval_rate"] == pytest.approx(2 / 3)
    assert result["bad_rate"] == pytest.approx(0.5)


def test_expected_profit_no_approvals_returns_zero():
    result = compute_expected_profit(
        y_pred=np.array([0.8, 0.9]),
        y_true=np.array([0, 1]),
        decision_mask=np.array([False, False]),
        loan_amounts=np.array([10000, 20000]),
    )

    assert result == {"total_profit": 0.0, "profit_per_loan": 0.0, "approval_rate": 0.0, "bad_rate": 0.0}


def test_oracle_profit_is_at_least_random_profit_on_ranked_data():
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 70 + [1] * 30)
    y_pred = np.concatenate([np.linspace(0.01, 0.2, 70), np.linspace(0.6, 0.95, 30)])
    loan_amounts = rng.uniform(5000, 25000, len(y_true))

    oracle = compute_oracle_profit(y_pred, y_true, loan_amounts)
    random = compute_random_profit(y_true, loan_amounts, approval_rate=oracle["approval_rate"], random_state=42)

    assert oracle["total_profit"] >= random["total_profit"]
    assert 0 <= oracle["approval_rate"] <= 1


def test_historical_profit_uses_high_scores_as_better_policy():
    y_true = np.array([0] * 60 + [1] * 40)
    loan_amounts = np.full(len(y_true), 10000.0)
    historical_scores = np.concatenate([np.linspace(0.8, 1.0, 60), np.linspace(0.0, 0.2, 40)])

    result = compute_historical_profit(y_true, loan_amounts, historical_scores)

    assert result["approval_rate"] > 0
    assert result["bad_rate"] < y_true.mean()


def test_oracle_profit_ratio_scales_between_random_and_oracle():
    ratio = compute_oracle_profit_ratio(model_profit=70.0, oracle_profit=100.0, random_profit=40.0)

    assert ratio == pytest.approx(0.5)
    assert compute_oracle_profit_ratio(10.0, 10.0, 10.0) == 0.0


def test_profit_frontier_returns_parameter_grid():
    y_true = np.array([0] * 50 + [1] * 50)
    y_pred = np.linspace(0.01, 0.99, 100)
    loan_amounts = np.full(100, 10000.0)

    frontier = compute_profit_frontier(
        y_pred,
        y_true,
        loan_amounts,
        lgd_values=[0.45, 0.60],
        funding_costs=[0.04],
        servicing_costs=[0.0, 0.01],
        prepayment_haircuts=[1.0],
        thresholds=[0.2, 0.5],
    )

    assert len(frontier) == 8
    assert {"lgd", "funding_cost", "servicing_cost", "prepayment_haircut", "threshold", "total_profit"}.issubset(frontier[0])


def test_profit_functions_reject_invalid_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_expected_profit(
            y_pred=np.array([0.1, 0.2]),
            y_true=np.array([0]),
            decision_mask=np.array([True, False]),
            loan_amounts=np.array([10000, 20000]),
        )
