from __future__ import annotations

import numpy as np


ZERO_PROFIT = {"total_profit": 0.0, "profit_per_loan": 0.0, "approval_rate": 0.0, "bad_rate": 0.0}


def compute_expected_profit(
    y_pred: np.ndarray | None,
    y_true: np.ndarray,
    decision_mask: np.ndarray,
    loan_amounts: np.ndarray,
    lgd: float = 0.45,
    interest_rate: float = 0.10,
    funding_cost: float = 0.04,
    servicing_cost: float = 0.0,
    prepayment_haircut: float = 1.0,
    term_years: float = 3.0,
) -> dict[str, float]:
    """Single-loan-period profit approximation for approved applications."""
    y_true, decision_mask, loan_amounts = _validate_profit_inputs(y_true, decision_mask, loan_amounts)
    _validate_optional_scores(y_pred, len(y_true))
    lgd = _validate_rate("lgd", lgd)
    interest_rate = _validate_non_negative("interest_rate", interest_rate)
    funding_cost = _validate_non_negative("funding_cost", funding_cost)
    servicing_cost = _validate_non_negative("servicing_cost", servicing_cost)
    prepayment_haircut = _validate_rate("prepayment_haircut", prepayment_haircut)
    term_years = _validate_non_negative("term_years", term_years)

    if decision_mask.sum() == 0:
        return ZERO_PROFIT.copy()

    principal = loan_amounts[decision_mask]
    effective_term = term_years * prepayment_haircut
    interest_income = principal * interest_rate * effective_term
    funding_expense = principal * funding_cost * effective_term
    servicing_expense = principal * servicing_cost
    losses = principal[y_true[decision_mask].astype(bool)] * lgd
    total_profit = interest_income.sum() - funding_expense.sum() - servicing_expense.sum() - losses.sum()

    return {
        "total_profit": float(total_profit),
        "profit_per_loan": float(total_profit / decision_mask.sum()),
        "approval_rate": float(decision_mask.mean()),
        "bad_rate": float(y_true[decision_mask].mean()),
    }


def compute_oracle_profit(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    loan_amounts: np.ndarray,
    **profit_kwargs,
) -> dict[str, float]:
    """Perfect-label benchmark: approve the best true outcomes for model-implied coverage levels."""
    y_true, y_pred, loan_amounts = _validate_score_profit_inputs(y_pred, y_true, loan_amounts)
    approval_counts = _approval_counts_from_scores(y_pred)
    order = np.lexsort((y_pred, y_true))
    return _best_profit_for_counts(y_pred, y_true, loan_amounts, order, approval_counts, **profit_kwargs)


def compute_random_profit(
    y_true: np.ndarray,
    loan_amounts: np.ndarray,
    approval_rate: float = 0.3,
    random_state: int | None = 42,
    **profit_kwargs,
) -> dict[str, float]:
    """Random approval baseline for the same simplified profit formula."""
    y_true = _as_binary_array("y_true", y_true)
    loan_amounts = _as_non_negative_array("loan_amounts", loan_amounts)
    if len(y_true) != len(loan_amounts):
        raise ValueError("y_true and loan_amounts must have the same length.")
    approval_rate = _validate_rate("approval_rate", approval_rate)

    rng = np.random.default_rng(random_state)
    approved = rng.random(len(y_true)) < approval_rate
    return compute_expected_profit(None, y_true, approved, loan_amounts, **profit_kwargs)


def compute_historical_profit(
    y_true: np.ndarray,
    loan_amounts: np.ndarray,
    historical_scores: np.ndarray,
    **profit_kwargs,
) -> dict[str, float]:
    """Historical policy proxy: higher historical score is treated as safer/better."""
    y_true = _as_binary_array("y_true", y_true)
    loan_amounts = _as_non_negative_array("loan_amounts", loan_amounts)
    historical_scores = _as_finite_array("historical_scores", historical_scores)
    if len(y_true) != len(loan_amounts) or len(y_true) != len(historical_scores):
        raise ValueError("y_true, loan_amounts, and historical_scores must have the same length.")

    thresholds = np.percentile(historical_scores, np.linspace(10, 90, 30))
    best = ZERO_PROFIT.copy()
    best_profit = -float("inf")
    for theta in thresholds:
        approved = historical_scores >= theta
        result = compute_expected_profit(None, y_true, approved, loan_amounts, **profit_kwargs)
        if result["total_profit"] > best_profit:
            best_profit = result["total_profit"]
            best = result
    return best


def compute_oracle_profit_ratio(model_profit: float, oracle_profit: float, random_profit: float) -> float:
    denominator = oracle_profit - random_profit
    if np.isclose(denominator, 0.0):
        return 0.0
    return float((model_profit - random_profit) / denominator)


def compute_profit_frontier(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    loan_amounts: np.ndarray,
    lgd_values: list[float] | None = None,
    funding_costs: list[float] | None = None,
    servicing_costs: list[float] | None = None,
    prepayment_haircuts: list[float] | None = None,
    thresholds: list[float] | np.ndarray | None = None,
    interest_rate: float = 0.10,
    term_years: float = 3.0,
) -> list[dict[str, float]]:
    """Profit frontier over credit and cost sensitivity parameters."""
    y_true, y_pred, loan_amounts = _validate_score_profit_inputs(y_pred, y_true, loan_amounts)
    lgd_values = lgd_values or [0.20, 0.35, 0.45, 0.60, 0.75, 0.90]
    funding_costs = funding_costs or [0.02, 0.04, 0.06, 0.08]
    servicing_costs = servicing_costs or [0.0, 0.01, 0.02]
    prepayment_haircuts = prepayment_haircuts or [0.5, 0.75, 1.0]
    thresholds = np.asarray(thresholds if thresholds is not None else np.percentile(y_pred, np.linspace(10, 90, 30)), dtype=float)

    results = []
    for lgd in lgd_values:
        for funding_cost in funding_costs:
            for servicing_cost in servicing_costs:
                for prepayment_haircut in prepayment_haircuts:
                    for threshold in thresholds:
                        approved = y_pred <= threshold
                        profit = compute_expected_profit(
                            y_pred,
                            y_true,
                            approved,
                            loan_amounts,
                            lgd=lgd,
                            interest_rate=interest_rate,
                            funding_cost=funding_cost,
                            servicing_cost=servicing_cost,
                            prepayment_haircut=prepayment_haircut,
                            term_years=term_years,
                        )
                        results.append(
                            {
                                "lgd": float(lgd),
                                "funding_cost": float(funding_cost),
                                "servicing_cost": float(servicing_cost),
                                "prepayment_haircut": float(prepayment_haircut),
                                "threshold": float(threshold),
                                **profit,
                            }
                        )
    return results


def _approval_counts_from_scores(y_pred: np.ndarray) -> np.ndarray:
    thresholds = np.percentile(y_pred, np.linspace(10, 90, 30))
    counts = [int((y_pred <= threshold).sum()) for threshold in thresholds]
    return np.unique(np.clip(np.array([0, *counts, len(y_pred)]), 0, len(y_pred)))


def _best_profit_for_counts(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    loan_amounts: np.ndarray,
    order: np.ndarray,
    approval_counts: np.ndarray,
    **profit_kwargs,
) -> dict[str, float]:
    best = ZERO_PROFIT.copy()
    best_profit = -float("inf")
    for n_approve in approval_counts:
        approved = np.zeros(len(y_true), dtype=bool)
        if n_approve > 0:
            approved[order[:n_approve]] = True
        result = compute_expected_profit(y_pred, y_true, approved, loan_amounts, **profit_kwargs)
        if result["total_profit"] > best_profit:
            best_profit = result["total_profit"]
            best = result
    return best


def _validate_score_profit_inputs(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    loan_amounts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = _as_binary_array("y_true", y_true)
    y_pred = _as_probability_array("y_pred", y_pred)
    loan_amounts = _as_non_negative_array("loan_amounts", loan_amounts)
    if len(y_true) != len(y_pred) or len(y_true) != len(loan_amounts):
        raise ValueError("y_pred, y_true, and loan_amounts must have the same length.")
    return y_true, y_pred, loan_amounts


def _validate_profit_inputs(
    y_true: np.ndarray,
    decision_mask: np.ndarray,
    loan_amounts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = _as_binary_array("y_true", y_true)
    decision_mask = np.asarray(decision_mask, dtype=bool)
    loan_amounts = _as_non_negative_array("loan_amounts", loan_amounts)
    if len(y_true) != len(decision_mask) or len(y_true) != len(loan_amounts):
        raise ValueError("y_true, decision_mask, and loan_amounts must have the same length.")
    return y_true, decision_mask, loan_amounts


def _validate_optional_scores(y_pred: np.ndarray | None, expected_length: int) -> None:
    if y_pred is None:
        return
    y_pred = _as_probability_array("y_pred", y_pred)
    if len(y_pred) != expected_length:
        raise ValueError("y_pred and y_true must have the same length.")


def _as_binary_array(name: str, values: np.ndarray) -> np.ndarray:
    array = _as_finite_array(name, values)
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must contain binary 0/1 labels.")
    return array.astype(int)


def _as_probability_array(name: str, values: np.ndarray) -> np.ndarray:
    array = _as_finite_array(name, values)
    if np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must be normalized to [0, 1].")
    return array


def _as_non_negative_array(name: str, values: np.ndarray) -> np.ndarray:
    array = _as_finite_array(name, values)
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative.")
    return array


def _as_finite_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _validate_rate(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"{name} must be in [0, 1].")
    return value


def _validate_non_negative(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value
