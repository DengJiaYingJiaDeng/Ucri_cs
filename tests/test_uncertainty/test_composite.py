import numpy as np
import pandas as pd
import pytest

from src.uncertainty.composite import CompositeUncertainty


class DummyTeacher:
    def __init__(self, probabilities: np.ndarray):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def compute_uncertainty(self, x: pd.DataFrame) -> dict[str, np.ndarray]:
        probabilities = self.probabilities[: len(x)]
        entropy = -probabilities * np.log(np.clip(probabilities, 1e-10, 1.0)) - (
            1 - probabilities
        ) * np.log(np.clip(1 - probabilities, 1e-10, 1.0))
        return {
            "variance": np.linspace(0.0, 0.1, len(x)),
            "entropy": entropy,
            "margin": 1.0 - np.abs(2 * probabilities - 1),
        }

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.probabilities[: len(x)]


def test_composite_uncertainty_combines_components():
    rng = np.random.default_rng(42)
    uncertainty_components = {
        "variance": rng.uniform(0, 0.1, 100),
        "entropy": rng.uniform(0, 0.7, 100),
        "margin": rng.uniform(0, 1, 100),
        "distance": rng.exponential(1, 100),
    }
    composite = CompositeUncertainty(alpha=(0.25, 0.25, 0.25, 0.25))

    result = composite.compute(uncertainty_components)

    assert len(result) == 100
    assert np.all((result >= 0) & (result <= 1))


def test_composite_uncertainty_equal_weights_preserve_monotonic_components():
    components = {
        "variance": np.array([0.01, 0.05, 0.1]),
        "entropy": np.array([0.3, 0.5, 0.7]),
        "margin": np.array([0.2, 0.5, 0.8]),
        "distance": np.array([0.5, 1.0, 2.0]),
    }
    composite = CompositeUncertainty(alpha=(0.25, 0.25, 0.25, 0.25))

    result = composite.compute(components)

    assert np.all(np.diff(result) >= 0)


def test_composite_uncertainty_single_component_uses_component_rank():
    components = {
        "variance": np.array([0.01, 0.05, 0.1, 0.02]),
        "entropy": np.zeros(4),
        "margin": np.zeros(4),
        "distance": np.zeros(4),
    }
    composite = CompositeUncertainty(alpha=(1.0, 0.0, 0.0, 0.0))

    result = composite.compute(components)

    assert result[np.argmax(components["variance"])] == 1.0
    assert result[np.argmin(components["variance"])] == 0.0
    assert np.array_equal(np.argsort(result), np.argsort(components["variance"]))


def test_composite_uncertainty_constant_components_return_zero():
    components = {
        "variance": np.ones(5),
        "entropy": np.ones(5),
        "margin": np.ones(5),
        "distance": np.ones(5),
    }
    composite = CompositeUncertainty()

    result = composite.compute(components)

    assert np.all(result == 0.0)


def test_composite_uncertainty_requires_all_components():
    composite = CompositeUncertainty()

    with pytest.raises(KeyError, match="distance"):
        composite.compute({"variance": np.array([1]), "entropy": np.array([1]), "margin": np.array([1])})


def test_composite_uncertainty_requires_matching_lengths():
    components = {
        "variance": np.array([0.1, 0.2]),
        "entropy": np.array([0.1]),
        "margin": np.array([0.1, 0.2]),
        "distance": np.array([0.1, 0.2]),
    }
    composite = CompositeUncertainty()

    with pytest.raises(ValueError, match="same length"):
        composite.compute(components)


def test_composite_uncertainty_alpha_must_sum_to_one():
    with pytest.raises(ValueError, match="Alpha must sum to 1"):
        CompositeUncertainty(alpha=(0.5, 0.5, 0.5, 0.0))


def test_compute_from_teacher_uses_teacher_and_distance_components():
    x_train = pd.DataFrame(
        {
            "loan_amount": [10000, 11000, 12000, 13000, 14000],
            "dti": [10, 12, 14, 16, 18],
        }
    )
    x_test = pd.DataFrame(
        {
            "loan_amount": [10000, 25000, 40000],
            "dti": [10, 35, 50],
        }
    )
    teacher = DummyTeacher(np.array([0.1, 0.5, 0.9]))
    composite = CompositeUncertainty(alpha=(0.25, 0.25, 0.25, 0.25))

    result = composite.compute_from_teacher(x_test, x_train, teacher)

    assert result.shape == (len(x_test),)
    assert np.all((result >= 0) & (result <= 1))
    assert composite._ref_train_distances is not None


def test_compute_from_teacher_samples_reference_distance_distribution():
    x_train = pd.DataFrame(
        {
            "loan_amount": np.linspace(10000, 20000, 20),
            "dti": np.linspace(10, 30, 20),
        }
    )
    x_test = x_train.head(3).copy()
    teacher = DummyTeacher(np.array([0.1, 0.5, 0.9]))
    composite = CompositeUncertainty()
    composite.REFERENCE_DISTANCE_SAMPLE_SIZE = 5

    result = composite.compute_from_teacher(x_test, x_train, teacher)

    assert result.shape == (len(x_test),)
    assert composite._ref_train_distances is not None
    assert len(composite._ref_train_distances) == 5


def test_fit_alpha_learns_valid_simplex_weights():
    x_train = pd.DataFrame(
        {
            "loan_amount": np.linspace(10000, 20000, 20),
            "dti": np.linspace(10, 30, 20),
        }
    )
    x_sim_rej = pd.DataFrame(
        {
            "loan_amount": np.linspace(11000, 30000, 12),
            "dti": np.linspace(12, 40, 12),
        }
    )
    teacher_probs = np.array([0.05, 0.08, 0.1, 0.2, 0.35, 0.45, 0.55, 0.65, 0.8, 0.9, 0.92, 0.95])
    y_hidden = (teacher_probs >= 0.5).astype(int)
    teacher = DummyTeacher(teacher_probs)
    composite = CompositeUncertainty()

    result = composite.fit_alpha(x_sim_rej, x_train, teacher, y_hidden)

    assert result is composite
    assert len(composite.alpha) == 4
    assert sum(composite.alpha) == pytest.approx(1.0)
    assert all(weight >= 0 for weight in composite.alpha)
