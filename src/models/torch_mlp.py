from __future__ import annotations

from importlib.util import find_spec

import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier

from src.models.device import validate_device_type, validate_gpu_device_id


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Small binary MLP classifier with an sklearn-compatible interface.

    When PyTorch is unavailable, the estimator falls back to sklearn's
    MLPClassifier so optional neural-network tests and teacher ensembles still
    run in lightweight environments.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        max_iter: int = 25,
        batch_size: int = 4096,
        learning_rate: float = 1e-3,
        random_state: int = 42,
        pos_weight: float = 1.0,
        device_type: str = "cpu",
        gpu_device_id: int = 0,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.pos_weight = pos_weight
        self.device_type = device_type
        self.gpu_device_id = gpu_device_id

    def fit(self, X, y):
        x_array = self._as_float_matrix(X)
        y_array = self._as_binary_targets(y, expected_length=len(x_array))
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.classes_ = np.array([0, 1])
        if find_spec("torch") is None:
            return self._fit_sklearn_fallback(x_array, y_array)

        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        self.backend_ = "torch"
        self.device_ = self._resolve_device(torch)
        self.model_ = self._build_network(nn, x_array.shape[1]).to(self.device_)

        generator = torch.Generator()
        generator.manual_seed(int(self.random_state))
        torch.manual_seed(int(self.random_state))

        x_tensor = torch.as_tensor(x_array, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_array.reshape(-1, 1), dtype=torch.float32)
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=min(int(self.batch_size), len(dataset)),
            shuffle=True,
            generator=generator,
        )

        positive_weight = max(float(self.pos_weight), 1e-6)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=self.device_))
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=float(self.learning_rate))

        self.model_.train()
        for _ in range(int(self.max_iter)):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device_)
                batch_y = batch_y.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not hasattr(self, "backend_"):
            raise ValueError("TorchMLPClassifier must be fitted before predicting.")
        if self.backend_ == "sklearn":
            return self._predict_sklearn_fallback_proba(X)

        import torch

        x_array = self._as_float_matrix(X)
        self.model_.eval()
        probabilities = []
        batch_size = max(1, int(self.batch_size))
        with torch.no_grad():
            for start in range(0, len(x_array), batch_size):
                batch = torch.as_tensor(x_array[start : start + batch_size], dtype=torch.float32, device=self.device_)
                logits = self.model_(batch).detach().cpu().numpy().reshape(-1)
                probabilities.append(expit(logits))
        positive = np.clip(np.concatenate(probabilities), 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def _fit_sklearn_fallback(self, x_array: np.ndarray, y_array: np.ndarray):
        self.backend_ = "sklearn"
        self.fallback_model_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=int(self.max_iter),
            batch_size=min(int(self.batch_size), len(x_array)),
            learning_rate_init=float(self.learning_rate),
            random_state=int(self.random_state),
            early_stopping=False,
        )
        self.fallback_model_.fit(x_array, y_array.astype(int))
        return self

    def _predict_sklearn_fallback_proba(self, X) -> np.ndarray:
        probabilities = np.asarray(self.fallback_model_.predict_proba(self._as_float_matrix(X)), dtype=float)
        if probabilities.shape[1] == 1:
            positive_class_index = int(self.fallback_model_.classes_[0])
            positive = np.ones(len(probabilities)) if positive_class_index == 1 else np.zeros(len(probabilities))
            return np.column_stack([1.0 - positive, positive])

        positive_column = int(np.flatnonzero(self.fallback_model_.classes_ == 1)[0])
        positive = np.clip(probabilities[:, positive_column], 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def _build_network(self, nn, n_features: int):
        layers = []
        input_dim = int(n_features)
        for hidden_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            input_dim = int(hidden_dim)
        layers.append(nn.Linear(input_dim, 1))
        return nn.Sequential(*layers)

    def _resolve_device(self, torch):
        device_type = validate_device_type(self.device_type)
        gpu_device_id = validate_gpu_device_id(self.gpu_device_id)
        if device_type == "gpu" and torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_device_id}")
        return torch.device("cpu")

    def _as_float_matrix(self, X) -> np.ndarray:
        if sparse.issparse(X):
            array = X.toarray()
        else:
            array = np.asarray(X)
        if array.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix.")
        if len(array) == 0:
            raise ValueError("X must not be empty.")
        array = np.asarray(array, dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise ValueError("X must contain finite values.")
        return array

    def _as_binary_targets(self, y, expected_length: int) -> np.ndarray:
        targets = np.asarray(y, dtype=float)
        if targets.ndim != 1:
            raise ValueError("y must be a one-dimensional array.")
        if len(targets) != expected_length:
            raise ValueError("X and y must have the same length.")
        if not np.isin(targets, [0, 1]).all():
            raise ValueError("y must contain binary 0/1 labels.")
        return targets.astype(np.float32)
