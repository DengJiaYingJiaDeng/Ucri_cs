import numpy as np
import pytest

from src.calibration.temperature import apply_temperature, fit_temperature, negative_log_likelihood


def test_fit_temperature_returns_positive_finite_value():
    logits = np.array([-3.0, -1.0, 0.2, 1.5, 3.0])
    y = np.array([0, 0, 1, 1, 1])

    temperature = fit_temperature(logits, y, coarse_grid_size=12, fine_grid_size=12)

    assert np.isfinite(temperature)
    assert temperature > 0


def test_temperature_scaling_outputs_probabilities_and_nll():
    logits = np.array([-2.0, 0.0, 2.0])
    y = np.array([0, 1, 1])

    probabilities = apply_temperature(logits, temperature=2.0)
    nll = negative_log_likelihood(logits, y, temperature=2.0)

    assert probabilities.shape == logits.shape
    assert np.all((probabilities > 0) & (probabilities < 1))
    assert np.isfinite(nll)


@pytest.mark.parametrize(
    "logits,y,error",
    [
        ([0.0, 1.0], [0], "same length"),
        ([0.0, 1.0], [0, 2], "binary"),
        ([[0.0, 1.0]], [0, 1], "one-dimensional"),
    ],
)
def test_temperature_validation_rejects_invalid_inputs(logits, y, error):
    with pytest.raises(ValueError, match=error):
        fit_temperature(np.asarray(logits), np.asarray(y))


def test_temperature_validation_rejects_invalid_temperature():
    with pytest.raises(ValueError, match="positive finite"):
        apply_temperature(np.array([0.0, 1.0]), temperature=0.0)
