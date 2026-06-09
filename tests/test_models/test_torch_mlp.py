import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

import src.models.torch_mlp as torch_mlp_module
from src.models.torch_mlp import TorchMLPClassifier


def test_torch_mlp_classifier_fit_predict_proba():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = TorchMLPClassifier(hidden_layer_sizes=(8,), max_iter=3, batch_size=16, random_state=7)

    _fit_ignoring_convergence_warning(model, X, y)
    probabilities = model.predict_proba(X[:10])

    assert probabilities.shape == (10, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_torch_mlp_classifier_falls_back_when_torch_missing(monkeypatch):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    original_find_spec = torch_mlp_module.find_spec

    def missing_torch(package_name):
        if package_name == "torch":
            return None
        return original_find_spec(package_name)

    monkeypatch.setattr(torch_mlp_module, "find_spec", missing_torch)
    model = TorchMLPClassifier(hidden_layer_sizes=(8,), max_iter=3, batch_size=16, random_state=7)

    _fit_ignoring_convergence_warning(model, X, y)
    probabilities = model.predict_proba(X[:10])

    assert model.backend_ == "sklearn"
    assert probabilities.shape == (10, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def _fit_ignoring_convergence_warning(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return model.fit(X, y)
