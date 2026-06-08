import numpy as np

from src.models.torch_mlp import TorchMLPClassifier


def test_torch_mlp_classifier_fit_predict_proba():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = TorchMLPClassifier(hidden_layer_sizes=(8,), max_iter=3, batch_size=16, random_state=7)

    model.fit(X, y)
    probabilities = model.predict_proba(X[:10])

    assert probabilities.shape == (10, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
