from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class PropensityModel:
    """Approval propensity model e(x)=P(accepted=1|x)."""

    def __init__(self, model_type: str = "logistic", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model: Pipeline | None = None
        self.backend_: str | None = None

    def _build_preprocessor(self, x: pd.DataFrame) -> ColumnTransformer:
        numeric_columns = x.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = [column for column in x.columns if column not in numeric_columns]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_columns),
                ("categorical", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
        )
        return preprocessor.set_output(transform="pandas")

    def _build_estimator(self):
        if self.model_type == "logistic":
            self.backend_ = "sklearn_logistic"
            return LogisticRegression(C=1.0, max_iter=2000, random_state=self.random_state, solver="liblinear")

        if self.model_type == "lightgbm":
            if find_spec("lightgbm") is not None:
                from lightgbm import LGBMClassifier

                self.backend_ = "lightgbm"
                return LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state,
                    n_jobs=1,
                    verbose=-1,
                )
            self.backend_ = "sklearn_hist_gradient_boosting"
            return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, random_state=self.random_state)

        if self.model_type == "catboost":
            if find_spec("catboost") is not None:
                from catboost import CatBoostClassifier

                self.backend_ = "catboost"
                return CatBoostClassifier(
                    iterations=100,
                    depth=5,
                    random_seed=self.random_state,
                    silent=True,
                    allow_writing_files=False,
                )
            self.backend_ = "sklearn_hist_gradient_boosting"
            return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, random_state=self.random_state)

        raise ValueError(f"Unknown propensity model type: {self.model_type}")

    def fit(self, x: pd.DataFrame, accepted: np.ndarray) -> PropensityModel:
        """Fit the approval propensity model."""
        x = pd.DataFrame(x).copy()
        accepted = np.asarray(accepted)
        if len(x) != len(accepted):
            raise ValueError("x and accepted must have the same length.")

        self.model = Pipeline(
            steps=[
                ("preprocessor", self._build_preprocessor(x)),
                ("estimator", self._build_estimator()),
            ]
        )
        self.model.fit(x, accepted)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Predict clipped approval probabilities."""
        if self.model is None:
            raise NotFittedError("PropensityModel must be fitted before calling predict_proba.")

        probabilities = self.model.predict_proba(pd.DataFrame(x))[:, 1]
        return np.clip(probabilities, 0.01, 0.99)

    def compute_weights(self, x: pd.DataFrame, eps: float = 0.01) -> np.ndarray:
        """Compute inverse propensity weights 1 / max(e(x), eps)."""
        if eps <= 0:
            raise ValueError("eps must be positive.")
        propensity = self.predict_proba(x)
        return 1.0 / np.maximum(propensity, eps)
