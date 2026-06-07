import numpy as np
import pandas as pd
import pytest

from src.calibration.cross_population import (
    cross_population_calibration_check,
    low_variance_high_error_diagnostic,
)
from src.evaluation.metrics import compute_brier


class DummyTeacher:
    def __init__(
        self,
        accepted_probs: np.ndarray,
        rejected_probs: np.ndarray,
        variances: np.ndarray | None = None,
        calibrated: bool = False,
        accepted_calibrated_probs: np.ndarray | None = None,
        rejected_calibrated_probs: np.ndarray | None = None,
        two_dimensional: bool = False,
    ):
        self.accepted_probs = np.asarray(accepted_probs, dtype=float)
        self.rejected_probs = np.asarray(rejected_probs, dtype=float)
        self.variances = None if variances is None else np.asarray(variances, dtype=float)
        self.calibrated = calibrated
        self.accepted_calibrated_probs = (
            self.accepted_probs
            if accepted_calibrated_probs is None
            else np.asarray(accepted_calibrated_probs, dtype=float)
        )
        self.rejected_calibrated_probs = (
            self.rejected_probs
            if rejected_calibrated_probs is None
            else np.asarray(rejected_calibrated_probs, dtype=float)
        )
        self.two_dimensional = two_dimensional

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        probabilities = self._select(x, self.accepted_probs, self.rejected_probs)
        if self.two_dimensional:
            return np.column_stack([1.0 - probabilities, probabilities])
        return probabilities

    def predict_calibrated(self, x: pd.DataFrame) -> np.ndarray:
        return self._select(x, self.accepted_calibrated_probs, self.rejected_calibrated_probs)

    def compute_uncertainty(self, x: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.variances is None:
            return {}
        return {"variance": self.variances[: len(x)]}

    def _select(self, x: pd.DataFrame, accepted_values: np.ndarray, rejected_values: np.ndarray) -> np.ndarray:
        if int(x["segment"].iloc[0]) == 0:
            return accepted_values[: len(x)]
        return rejected_values[: len(x)]


def test_cross_population_calibration_reports_rejected_like_gap():
    y_accepted = np.array([0, 0, 1, 1] * 8)
    y_rejected = np.array([1, 1, 0, 0] * 8)
    accepted_probs = np.array([0.15, 0.25, 0.75, 0.85] * 8)
    rejected_probs = np.array([0.15, 0.25, 0.75, 0.85] * 8)
    teacher = DummyTeacher(
        accepted_probs=np.full(len(y_accepted), 0.5),
        rejected_probs=np.full(len(y_rejected), 0.5),
        calibrated=True,
        accepted_calibrated_probs=accepted_probs,
        rejected_calibrated_probs=rejected_probs,
    )

    result = cross_population_calibration_check(
        teacher,
        pd.DataFrame({"segment": np.zeros(len(y_accepted), dtype=int)}),
        y_accepted,
        pd.DataFrame({"segment": np.ones(len(y_rejected), dtype=int)}),
        y_rejected,
    )

    assert result["n_accepted"] == len(y_accepted)
    assert result["n_hidden_rejected"] == len(y_rejected)
    assert result["accepted_brier"] == pytest.approx(compute_brier(y_accepted, accepted_probs))
    assert result["rejected_like_brier"] > result["accepted_brier"]
    assert result["ece_gap"] > 0
    assert result["calib_slope_gap"] < 0
    assert "calib_intercept_gap" in result


def test_cross_population_calibration_accepts_sklearn_style_probabilities():
    y = np.array([0, 0, 1, 1] * 5)
    probs = np.array([0.1, 0.3, 0.7, 0.9] * 5)
    teacher = DummyTeacher(probs, probs, two_dimensional=True)

    result = cross_population_calibration_check(
        teacher,
        pd.DataFrame({"segment": np.zeros(len(y), dtype=int)}),
        y,
        pd.DataFrame({"segment": np.ones(len(y), dtype=int)}),
        y,
    )

    assert result["accepted_brier"] == pytest.approx(result["rejected_like_brier"])
    assert abs(result["brier_gap"]) < 1e-12


def test_low_variance_high_error_diagnostic_finds_confidently_wrong_region():
    y_rejected = np.array([0, 0, 1, 0, 0])
    rejected_probs = np.array([0.95, 0.90, 0.10, 0.40, 0.20])
    variances = np.array([0.01, 0.02, 0.03, 0.08, 0.10])
    teacher = DummyTeacher(
        accepted_probs=np.array([0.5]),
        rejected_probs=rejected_probs,
        variances=variances,
    )

    result = low_variance_high_error_diagnostic(
        teacher,
        pd.DataFrame({"segment": np.ones(len(y_rejected), dtype=int)}),
        y_rejected,
        variance_pct=50.0,
        error_pct=40.0,
    )

    assert result["n_confidently_wrong"] == 3
    assert result["confidently_wrong_rate"] == pytest.approx(3 / 5)
    assert result["confidently_wrong_mean_error"] == pytest.approx(np.mean([0.95, 0.90, 0.90]))
    assert result["variance_threshold"] == pytest.approx(0.03)
    assert result["n_hidden_rejected"] == len(y_rejected)


def test_cross_population_calibration_validates_inputs():
    teacher = DummyTeacher(np.array([0.2, 0.8]), np.array([0.2, 0.8]))
    x = pd.DataFrame({"segment": [0, 0]})

    with pytest.raises(ValueError, match="same length"):
        cross_population_calibration_check(
            teacher,
            x,
            np.array([0, 1, 0]),
            pd.DataFrame({"segment": [1, 1]}),
            np.array([0, 1]),
        )

    with pytest.raises(ValueError, match="both classes"):
        cross_population_calibration_check(
            teacher,
            x,
            np.array([1, 1]),
            pd.DataFrame({"segment": [1, 1]}),
            np.array([0, 1]),
        )


def test_low_variance_high_error_diagnostic_validates_inputs():
    y = np.array([0, 1])
    x = pd.DataFrame({"segment": [1, 1]})
    teacher = DummyTeacher(np.array([0.5]), np.array([0.2, 0.8]))

    with pytest.raises(KeyError, match="variance"):
        low_variance_high_error_diagnostic(teacher, x, y)

    teacher_with_variance = DummyTeacher(np.array([0.5]), np.array([0.2, 0.8]), variances=np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="variance_pct"):
        low_variance_high_error_diagnostic(teacher_with_variance, x, y, variance_pct=101.0)
