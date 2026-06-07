import numpy as np
import pytest

from src.evaluation.statistics import (
    benjamini_hochberg,
    bootstrap_ci,
    cliffs_delta,
    compute_summary_stats,
    delong_test,
    holm_bonferroni,
    paired_standardized_mean_difference,
    paired_wilcoxon,
)


def test_bootstrap_ci_is_reproducible_and_contains_mean():
    values = np.array([0.68, 0.70, 0.71, 0.69, 0.72])

    ci_1 = bootstrap_ci(values, n_resamples=300, random_state=7)
    ci_2 = bootstrap_ci(values, n_resamples=300, random_state=7)

    assert ci_1 == ci_2
    assert ci_1[0] <= values.mean() <= ci_1[1]
    assert 0.68 <= ci_1[0] <= ci_1[1] <= 0.72


def test_bootstrap_ci_supports_custom_statistic():
    values = np.array([1.0, 2.0, 100.0])

    ci_low, ci_high = bootstrap_ci(values, n_resamples=200, random_state=3, statistic=np.median)

    assert ci_low <= ci_high
    assert 1.0 <= ci_low <= 100.0


def test_paired_wilcoxon_detects_directional_improvement():
    model_a = np.array([0.71, 0.72, 0.73, 0.75, 0.74, 0.76, 0.77, 0.78])
    model_b = np.array([0.65, 0.66, 0.68, 0.70, 0.69, 0.71, 0.72, 0.73])

    result = paired_wilcoxon(model_a, model_b, alternative="greater")

    assert result["statistic"] > 0
    assert result["p_value"] < 0.05


def test_paired_wilcoxon_equal_scores_returns_non_significant():
    scores = np.array([0.1, 0.2, 0.3])

    result = paired_wilcoxon(scores, scores)

    assert result == {"statistic": 0.0, "p_value": 1.0}


def test_holm_bonferroni_step_down_correction():
    p_values = [0.001, 0.02, 0.03, 0.2]

    rejected = holm_bonferroni(p_values, alpha=0.05)

    assert rejected == [True, False, False, False]


def test_benjamini_hochberg_rejects_up_to_largest_passing_rank():
    p_values = [0.001, 0.02, 0.03, 0.2]

    rejected = benjamini_hochberg(p_values, alpha=0.05)

    assert rejected == [True, True, True, False]


def test_cliffs_delta_reports_dominance_direction():
    assert cliffs_delta(np.array([3, 4, 5]), np.array([0, 1, 2])) == 1.0
    assert cliffs_delta(np.array([0, 1, 2]), np.array([3, 4, 5])) == -1.0
    assert cliffs_delta(np.array([1, 2]), np.array([1, 2])) == 0.0


def test_paired_standardized_mean_difference_uses_seedwise_deltas():
    a = np.array([0.8, 0.9, 1.0, 1.2])
    b = np.array([0.7, 0.8, 0.9, 0.9])

    result = paired_standardized_mean_difference(a, b)

    assert result > 0


def test_delong_test_reports_auc_difference():
    y_true = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    reversed_scores = np.array([0.9, 0.8, 0.2, 0.1])

    result = delong_test(y_true, perfect, reversed_scores)

    assert result["auc_a"] == 1.0
    assert result["auc_b"] == 0.0
    assert result["diff"] == 1.0
    assert result["p_value"] == 0.0


def test_delong_test_equal_predictions_returns_p_one():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.8])

    result = delong_test(y_true, scores, scores)

    assert result["diff"] == 0.0
    assert result["p_value"] == 1.0


def test_compute_summary_stats_returns_ci_and_seed_count():
    summary = compute_summary_stats(
        {"AUROC": [0.70, 0.72, 0.71, 0.73]},
        "AUROC",
        n_bootstrap=200,
        random_state=9,
    )

    assert summary["metric"] == "AUROC"
    assert summary["n_seeds"] == 4
    assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"]
    assert summary["std"] > 0


def test_statistics_validate_inputs():
    with pytest.raises(ValueError, match="one-dimensional"):
        bootstrap_ci(np.ones((2, 2)))

    with pytest.raises(ValueError, match="same length"):
        paired_wilcoxon(np.array([1.0, 2.0]), np.array([1.0]))

    with pytest.raises(ValueError, match="p_values"):
        holm_bonferroni([0.1, 1.2])

    with pytest.raises(ValueError, match="both classes"):
        delong_test(np.array([1, 1, 1]), np.array([0.2, 0.3, 0.4]), np.array([0.2, 0.3, 0.4]))

    with pytest.raises(KeyError, match="Metric not found"):
        compute_summary_stats({"Brier": [0.1, 0.2]}, "AUROC")
