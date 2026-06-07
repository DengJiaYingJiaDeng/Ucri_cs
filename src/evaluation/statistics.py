from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import norm, wilcoxon


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
    statistic: Callable[[np.ndarray], float] | None = None,
) -> tuple[float, float]:
    """Bootstrap two-sided confidence interval for a metric across seeds."""
    values = _as_1d_finite_array("values", values)
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    alpha = _validate_alpha(alpha)
    statistic = statistic or (lambda sample: float(np.mean(sample)))

    rng = np.random.default_rng(random_state)
    estimates = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample_indices = rng.integers(0, len(values), size=len(values))
        estimates[index] = float(statistic(values[sample_indices]))

    lower = np.percentile(estimates, 100.0 * alpha / 2.0)
    upper = np.percentile(estimates, 100.0 * (1.0 - alpha / 2.0))
    return float(lower), float(upper)


def paired_wilcoxon(
    model_a_scores: np.ndarray,
    model_b_scores: np.ndarray,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test, the default confirmatory test."""
    a, b = _validate_paired_scores(model_a_scores, model_b_scores)
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be one of {'two-sided', 'less', 'greater'}.")
    if np.allclose(a - b, 0.0):
        return {"statistic": 0.0, "p_value": 1.0}

    statistic, p_value = wilcoxon(a, b, zero_method="wilcox", correction=False, alternative=alternative)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def holm_bonferroni(p_values: list[float] | np.ndarray, alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni correction for confirmatory pairwise comparisons."""
    p_values = _validate_p_values(p_values)
    alpha = _validate_alpha(alpha)
    order = np.argsort(p_values, kind="mergesort")
    rejected = np.zeros(len(p_values), dtype=bool)

    for rank, original_index in enumerate(order):
        threshold = alpha / (len(p_values) - rank)
        if p_values[original_index] <= threshold:
            rejected[original_index] = True
        else:
            break
    return rejected.tolist()


def benjamini_hochberg(p_values: list[float] | np.ndarray, alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR control for exploratory large tables."""
    p_values = _validate_p_values(p_values)
    alpha = _validate_alpha(alpha)
    order = np.argsort(p_values, kind="mergesort")
    sorted_p = p_values[order]
    thresholds = alpha * (np.arange(1, len(p_values) + 1) / len(p_values))
    passing = sorted_p <= thresholds

    rejected = np.zeros(len(p_values), dtype=bool)
    if passing.any():
        max_rank = int(np.where(passing)[0].max())
        rejected[order[: max_rank + 1]] = True
    return rejected.tolist()


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta dominance effect size."""
    a = _as_1d_finite_array("a", a)
    b = _as_1d_finite_array("b", b)
    dominance = 0.0
    for value in a:
        dominance += float((value > b).sum() - (value < b).sum())
    return float(dominance / (len(a) * len(b)))


def paired_standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Paired standardized mean difference for seed-wise score deltas."""
    a, b = _validate_paired_scores(a, b)
    diff = a - b
    std = float(np.std(diff, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(diff) / std)


def delong_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict[str, float]:
    """Supplementary DeLong-style paired AUROC comparison.

    Wilcoxon across seeds remains the default project test; DeLong is reported
    only as an AUROC-specific supplement.
    """
    y_true, pred_a, pred_b = _validate_auc_inputs(y_true, y_pred_a, y_pred_b)
    auc_a, v10_a, v01_a = _auc_structural_components(y_true, pred_a)
    auc_b, v10_b, v01_b = _auc_structural_components(y_true, pred_b)

    positives = y_true == 1
    negatives = y_true == 0
    sx = _covariance_2x2(v10_a, v10_b)
    sy = _covariance_2x2(v01_a, v01_b)
    n_pos = max(int(positives.sum()), 1)
    n_neg = max(int(negatives.sum()), 1)
    cov00 = sx[0] / n_pos + sy[0] / n_neg
    cov11 = sx[1] / n_pos + sy[1] / n_neg
    cov01 = sx[2] / n_pos + sy[2] / n_neg
    variance = float(cov00 + cov11 - 2.0 * cov01)
    diff = float(auc_a - auc_b)

    if variance <= 1e-15:
        p_value = 1.0 if abs(diff) <= 1e-12 else 0.0
        z_score = 0.0 if abs(diff) <= 1e-12 else float(np.sign(diff) * np.inf)
    else:
        z_score = float(diff / np.sqrt(variance))
        p_value = float(2.0 * (1.0 - norm.cdf(abs(z_score))))

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "diff": diff,
        "z": z_score,
        "p_value": float(np.clip(p_value, 0.0, 1.0)),
    }


def compute_summary_stats(
    results_by_seed: dict[str, list[float] | np.ndarray],
    metric_name: str,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, float | int | str]:
    """Compute mean, std, and bootstrap CI for one metric across seeds."""
    if metric_name not in results_by_seed:
        raise KeyError(f"Metric not found: {metric_name}")
    values = _as_1d_finite_array(metric_name, results_by_seed[metric_name])
    ci_low, ci_high = bootstrap_ci(
        values,
        n_resamples=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )
    return {
        "metric": metric_name,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_seeds": int(len(values)),
    }


def _auc_structural_components(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    positive_scores = y_pred[y_true == 1]
    negative_scores = y_pred[y_true == 0]
    comparison = _pairwise_auc_comparison(positive_scores, negative_scores)
    v10 = comparison.mean(axis=1)
    v01 = comparison.mean(axis=0)
    return float(comparison.mean()), v10, v01


def _pairwise_auc_comparison(positive_scores: np.ndarray, negative_scores: np.ndarray) -> np.ndarray:
    diff = positive_scores[:, None] - negative_scores[None, :]
    return (diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)


def _covariance_2x2(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    if len(a) <= 1:
        return 0.0, 0.0, 0.0
    a_centered = a - float(np.mean(a))
    b_centered = b - float(np.mean(b))
    denominator = len(a) - 1
    var_a = float(np.sum(a_centered * a_centered) / denominator)
    var_b = float(np.sum(b_centered * b_centered) / denominator)
    cov_ab = float(np.sum(a_centered * b_centered) / denominator)
    return var_a, var_b, cov_ab


def _validate_auc_inputs(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    pred_a = _as_1d_finite_array("y_pred_a", y_pred_a)
    pred_b = _as_1d_finite_array("y_pred_b", y_pred_b)
    if y_true.ndim != 1:
        raise ValueError("y_true must be a one-dimensional array.")
    if len(y_true) != len(pred_a) or len(y_true) != len(pred_b):
        raise ValueError("y_true, y_pred_a, and y_pred_b must have the same length.")
    if len(y_true) == 0:
        raise ValueError("AUC inputs must not be empty.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain binary 0/1 labels.")
    if len(np.unique(y_true)) != 2:
        raise ValueError("y_true must contain both classes for DeLong test.")
    return y_true.astype(int), np.clip(pred_a, 0.0, 1.0), np.clip(pred_b, 0.0, 1.0)


def _validate_paired_scores(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = _as_1d_finite_array("model_a_scores", a)
    b = _as_1d_finite_array("model_b_scores", b)
    if len(a) != len(b):
        raise ValueError("model_a_scores and model_b_scores must have the same length.")
    return a, b


def _validate_p_values(p_values: list[float] | np.ndarray) -> np.ndarray:
    values = _as_1d_finite_array("p_values", np.asarray(p_values, dtype=float))
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be in [0, 1].")
    return values


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be in (0, 1).")
    return alpha


def _as_1d_finite_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array
