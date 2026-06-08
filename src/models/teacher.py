from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data.encoding import CategoricalStringifier
from src.models.device import catboost_device_params, lightgbm_device_params, validate_device_type, validate_gpu_device_id
from src.models.sklearn_compat import predict_proba_silencing_lightgbm_feature_name_warning


class TeacherEnsemble:
    """Multi-model PD teacher with ensemble uncertainty and temperature calibration."""

    def __init__(
        self,
        n_models: int = 5,
        model_types: list[str] | None = None,
        random_state: int = 42,
        class_weight: str = "balanced",
        scale_pos_weight: float | None = None,
        device_type: str = "cpu",
        gpu_device_id: int = 0,
    ):
        self.n_models = n_models
        default_model_cycle = ["lightgbm", "catboost", "mlp"]
        self.model_types = model_types or [default_model_cycle[i % len(default_model_cycle)] for i in range(n_models)]
        if len(self.model_types) != n_models:
            raise ValueError("model_types length must match n_models.")
        self.random_state = random_state
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight
        self.device_type = validate_device_type(device_type)
        self.gpu_device_id = validate_gpu_device_id(gpu_device_id)
        self.models: list[Pipeline] = []
        self.temperature = 1.0
        self.calibrated = False

    def _compute_pos_weight(self, y: np.ndarray) -> float:
        y = np.asarray(y)
        n_negative = int((y == 0).sum())
        n_positive = int((y == 1).sum())
        if n_positive == 0:
            return 1.0
        return min(n_negative / n_positive, 20.0)

    def _build_preprocessor(self, x: pd.DataFrame, model_type: str) -> ColumnTransformer:
        numeric_columns = x.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = [column for column in x.columns if column not in numeric_columns]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        if self._use_ordinal_categories(model_type):
            categorical_pipeline = Pipeline(
                steps=[
                    ("stringifier", CategoricalStringifier()),
                    (
                        "ordinal",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                            encoded_missing_value=-1,
                        ),
                    ),
                ]
            )
        else:
            categorical_pipeline = Pipeline(
                steps=[
                    ("stringifier", CategoricalStringifier()),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                ]
            )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_columns),
                ("categorical", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )

    def _use_ordinal_categories(self, model_type: str) -> bool:
        return model_type == "catboost" or (model_type == "lightgbm" and find_spec("lightgbm") is None)

    def _build_estimator(self, model_type: str, seed: int, pos_weight: float):
        if model_type == "lightgbm":
            if find_spec("lightgbm") is not None:
                from lightgbm import LGBMClassifier

                return LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=seed,
                    n_jobs=1,
                    verbose=-1,
                    scale_pos_weight=pos_weight,
                    **lightgbm_device_params(self.device_type, self.gpu_device_id),
                )
            return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, random_state=seed)

        if model_type == "catboost":
            if find_spec("catboost") is not None:
                from catboost import CatBoostClassifier

                return CatBoostClassifier(
                    iterations=100,
                    depth=5,
                    random_seed=seed,
                    silent=True,
                    scale_pos_weight=pos_weight,
                    allow_writing_files=False,
                    **catboost_device_params(self.device_type, self.gpu_device_id),
                )
            return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, random_state=seed)

        if model_type == "mlp":
            return LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=seed,
                solver="liblinear",
            )

        raise ValueError(f"Unknown teacher model type: {model_type}")

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> TeacherEnsemble:
        x = pd.DataFrame(x).copy()
        y = np.asarray(y)
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")

        pos_weight = self.scale_pos_weight if self.scale_pos_weight is not None else self._compute_pos_weight(y)
        self.scale_pos_weight = pos_weight
        self.models = []
        rng = np.random.default_rng(self.random_state)

        for index, model_type in enumerate(self.model_types):
            seed = self.random_state + index
            model = Pipeline(
                steps=[
                    ("preprocessor", self._build_preprocessor(x, model_type)),
                    ("estimator", self._build_estimator(model_type, seed, pos_weight)),
                ]
            )
            sample_indices = rng.choice(len(x), size=len(x), replace=True)
            model.fit(x.iloc[sample_indices], y[sample_indices])
            self.models.append(model)

        self.calibrated = False
        self.temperature = 1.0
        return self

    def predict_individual(self, x: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise NotFittedError("TeacherEnsemble must be fitted before predicting.")
        return np.column_stack(
            [
                predict_proba_silencing_lightgbm_feature_name_warning(model, pd.DataFrame(x))[:, 1]
                for model in self.models
            ]
        )

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self.predict_individual(x).mean(axis=1)

    def compute_uncertainty(self, x: pd.DataFrame) -> dict[str, np.ndarray]:
        individual_predictions = self.predict_individual(x)
        mean_prediction = individual_predictions.mean(axis=1)
        clipped_mean = np.clip(mean_prediction, 1e-10, 1 - 1e-10)

        variance = individual_predictions.var(axis=1)
        entropy = -clipped_mean * np.log(clipped_mean) - (1 - clipped_mean) * np.log(1 - clipped_mean)
        margin = 1.0 - np.abs(2 * mean_prediction - 1)
        return {
            "variance": variance,
            "entropy": entropy,
            "margin": margin,
            "mean": mean_prediction,
        }

    def calibrate(self, x: pd.DataFrame, y: np.ndarray, method: str = "temperature") -> TeacherEnsemble:
        if method != "temperature":
            raise ValueError(f"Unknown calibration method: {method}")
        logits = self._to_logits(self.predict_proba(x))
        self.temperature = self._fit_temperature(logits, np.asarray(y))
        self.calibrated = True
        return self

    def predict_calibrated(self, x: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(x)
        if not self.calibrated:
            return probabilities
        logits = self._to_logits(probabilities)
        return expit(logits / self.temperature)

    def _to_logits(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(probabilities, 1e-10, 1 - 1e-10)
        return np.log(clipped / (1 - clipped))

    def _fit_temperature(self, logits: np.ndarray, y: np.ndarray) -> float:
        def nll(temperature: float) -> float:
            probabilities = expit(logits / temperature)
            probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
            return float(-np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities)))

        coarse_grid = np.geomspace(0.01, 10.0, 80)
        coarse_scores = np.array([nll(float(temperature)) for temperature in coarse_grid])
        best_temperature = float(coarse_grid[int(np.argmin(coarse_scores))])

        lower = max(0.01, best_temperature / 1.5)
        upper = min(10.0, best_temperature * 1.5)
        fine_grid = np.linspace(lower, upper, 80)
        fine_scores = np.array([nll(float(temperature)) for temperature in fine_grid])
        return float(fine_grid[int(np.argmin(fine_scores))])
