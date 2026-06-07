from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _soft_bce_grad_hess(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and hessian for soft BCE with probability predictions."""
    probability = np.clip(np.asarray(pred, dtype=float), 1e-10, 1 - 1e-10)
    target = np.asarray(target, dtype=float)
    if probability.shape != target.shape:
        raise ValueError("pred and target must have the same shape.")

    grad = probability - target
    hess = probability * (1 - probability)
    return grad, hess


class StudentModel:
    """Lightweight student PD model trained on accepted labels and weighted soft pseudo-labels."""

    def __init__(
        self,
        model_type: str = "lightgbm",
        random_state: int = 42,
        scale_pos_weight: float | None = None,
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight
        self.model: Pipeline | None = None
        self.backend_: str | None = None
        self.temperature = 1.0
        self.calibrated = False
        self.training_targets_: np.ndarray | None = None
        self.training_sample_weights_: np.ndarray | None = None

    def _compute_pos_weight(self, y: np.ndarray) -> float:
        y = np.asarray(y)
        n_negative = int((y == 0).sum())
        n_positive = int((y == 1).sum())
        if n_positive == 0:
            return 1.0
        return min(n_negative / n_positive, 20.0)

    @staticmethod
    def tune_scale_pos_weight(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        model_type: str = "lightgbm",
        cap_values: list[float] | None = None,
    ) -> dict[str, object]:
        """Choose a positive-class weight cap on validation PR-AUC."""
        cap_values = cap_values or [5.0, 10.0, 20.0, 50.0]
        y_train = np.asarray(y_train)
        raw_weight = (y_train == 0).sum() / max(int((y_train == 1).sum()), 1)
        best_weight: float | None = None
        best_pr_auc = -np.inf
        results = []

        from src.evaluation.metrics import compute_all_metrics

        for cap in cap_values:
            pos_weight = float(min(raw_weight, cap))
            model = StudentModel(model_type=model_type, scale_pos_weight=pos_weight)
            model.fit(X_train, y_train)
            predictions = model.predict_proba(X_val)
            metrics = compute_all_metrics(np.asarray(y_val), predictions)
            results.append({"cap": float(cap), "pos_weight": pos_weight, **metrics})
            if metrics["PR-AUC"] > best_pr_auc:
                best_pr_auc = metrics["PR-AUC"]
                best_weight = pos_weight

        return {"best_pos_weight": best_weight, "tuning_results": results}

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

    def _build_estimator(self, pos_weight: float, use_internal_pos_weight: bool = True):
        effective_pos_weight = pos_weight if use_internal_pos_weight else 1.0

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
                    scale_pos_weight=effective_pos_weight,
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
                    scale_pos_weight=effective_pos_weight,
                    allow_writing_files=False,
                )
            self.backend_ = "sklearn_hist_gradient_boosting"
            return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, random_state=self.random_state)

        if self.model_type == "logistic":
            self.backend_ = "sklearn_logistic"
            class_weight = {0: 1.0, 1: effective_pos_weight} if use_internal_pos_weight else None
            return LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=self.random_state,
                class_weight=class_weight,
            )

        raise ValueError(f"Unknown student model type: {self.model_type}")

    def fit(
        self,
        X_labeled: pd.DataFrame,
        y_labeled: np.ndarray,
        X_unlabeled: pd.DataFrame | None = None,
        teacher_probs: np.ndarray | None = None,
        pseudo_weights: np.ndarray | None = None,
        lambda_distill: float = 0.3,
    ) -> StudentModel:
        x_labeled, y_labeled = self._validate_labeled_inputs(X_labeled, y_labeled)
        if self.scale_pos_weight is None:
            self.scale_pos_weight = self._compute_pos_weight(y_labeled)

        self.temperature = 1.0
        self.calibrated = False

        has_distillation = X_unlabeled is not None or teacher_probs is not None or pseudo_weights is not None
        if has_distillation:
            self._fit_with_distillation(
                x_labeled,
                y_labeled,
                X_unlabeled,
                teacher_probs,
                pseudo_weights,
                lambda_distill,
            )
        else:
            self.training_targets_ = y_labeled.astype(float)
            supervised_weights = np.ones(len(y_labeled), dtype=float)
            supervised_weights[y_labeled == 1] = self.scale_pos_weight
            self.training_sample_weights_ = supervised_weights
            self.model = self._build_pipeline(x_labeled, use_internal_pos_weight=True)
            self.model.fit(x_labeled, y_labeled)

        return self

    def _fit_with_distillation(
        self,
        x_labeled: pd.DataFrame,
        y_labeled: np.ndarray,
        X_unlabeled: pd.DataFrame | None,
        teacher_probs: np.ndarray | None,
        pseudo_weights: np.ndarray | None,
        lambda_distill: float,
    ) -> None:
        if X_unlabeled is None or teacher_probs is None:
            raise ValueError("X_unlabeled and teacher_probs must be provided together.")
        if lambda_distill < 0 or not np.isfinite(lambda_distill):
            raise ValueError("lambda_distill must be a non-negative finite value.")

        x_unlabeled = pd.DataFrame(X_unlabeled).copy().reindex(columns=x_labeled.columns)
        teacher_probs = self._as_1d_array("teacher_probs", teacher_probs)
        if len(x_unlabeled) != len(teacher_probs):
            raise ValueError("X_unlabeled, teacher_probs, and pseudo_weights must have the same length.")
        teacher_probs = np.clip(teacher_probs, 1e-6, 1 - 1e-6)

        if pseudo_weights is None:
            pseudo_weights = np.ones(len(x_unlabeled), dtype=float)
        else:
            pseudo_weights = self._as_1d_array("pseudo_weights", pseudo_weights)
            if len(pseudo_weights) != len(x_unlabeled):
                raise ValueError("X_unlabeled, teacher_probs, and pseudo_weights must have the same length.")
            if np.any(pseudo_weights < 0):
                raise ValueError("pseudo_weights must be non-negative.")

        x_all = pd.concat([x_labeled, x_unlabeled], ignore_index=True)
        soft_targets = np.concatenate([y_labeled.astype(float), teacher_probs])

        supervised_weights = np.ones(len(y_labeled), dtype=float)
        supervised_weights[y_labeled == 1] = self.scale_pos_weight
        distill_weights = pseudo_weights.astype(float) * lambda_distill
        sample_weights = np.concatenate([supervised_weights, distill_weights])

        self.training_targets_ = soft_targets.copy()
        self.training_sample_weights_ = sample_weights.copy()

        x_expanded, y_expanded, weights_expanded = self._expand_soft_targets(x_all, soft_targets, sample_weights)
        self.model = self._build_pipeline(x_expanded, use_internal_pos_weight=False)
        self.model.fit(x_expanded, y_expanded, estimator__sample_weight=weights_expanded)

    def _build_pipeline(self, x: pd.DataFrame, use_internal_pos_weight: bool) -> Pipeline:
        assert self.scale_pos_weight is not None
        return Pipeline(
            steps=[
                ("preprocessor", self._build_preprocessor(x)),
                ("estimator", self._build_estimator(self.scale_pos_weight, use_internal_pos_weight)),
            ]
        )

    def _expand_soft_targets(
        self,
        x: pd.DataFrame,
        soft_targets: np.ndarray,
        sample_weights: np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        positive_weights = sample_weights * soft_targets
        negative_weights = sample_weights * (1.0 - soft_targets)
        x_expanded = pd.concat([x, x], ignore_index=True)
        y_expanded = np.concatenate([np.ones(len(x), dtype=int), np.zeros(len(x), dtype=int)])
        weights_expanded = np.concatenate([positive_weights, negative_weights])
        nonzero = weights_expanded > 0
        return x_expanded.loc[nonzero].reset_index(drop=True), y_expanded[nonzero], weights_expanded[nonzero]

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        raw_probabilities = self._predict_raw_proba(x)
        if not self.calibrated:
            return raw_probabilities

        logits = self._to_logits(raw_probabilities)
        calibrated = expit(logits / self.temperature)
        return np.clip(calibrated, 0.0, 1.0)

    def post_calibrate(self, x: pd.DataFrame, y: np.ndarray, method: str = "temperature") -> dict[str, float]:
        if method != "temperature":
            raise ValueError(f"Unknown calibration method: {method}")
        y = self._validate_binary_labels(y, expected_length=len(pd.DataFrame(x)))
        raw_probabilities = self._predict_raw_proba(x)
        logits = self._to_logits(raw_probabilities)

        def nll(temperature_array: np.ndarray) -> float:
            temperature = float(temperature_array[0])
            probabilities = expit(logits / temperature)
            probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
            return float(-np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities)))

        result = minimize(nll, x0=np.array([1.0]), bounds=[(0.01, 10.0)], method="L-BFGS-B")
        self.temperature = float(result.x[0])
        self.calibrated = True

        from src.evaluation.metrics import compute_ece

        calibrated_probabilities = self.predict_proba(x)
        return {
            "before_ECE": compute_ece(y, raw_probabilities),
            "after_ECE": compute_ece(y, calibrated_probabilities),
            "temperature": self.temperature,
        }

    def _predict_raw_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise NotFittedError("StudentModel must be fitted before calling predict_proba.")
        probabilities = self.model.predict_proba(pd.DataFrame(x))[:, 1]
        return np.clip(probabilities, 0.0, 1.0)

    def _validate_labeled_inputs(self, x: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        x = pd.DataFrame(x).copy()
        y = self._validate_binary_labels(y, expected_length=len(x))
        return x, y

    def _validate_binary_labels(self, y: np.ndarray, expected_length: int) -> np.ndarray:
        y = self._as_1d_array("y", y).astype(int)
        if len(y) != expected_length:
            raise ValueError("x and y must have the same length.")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("y must contain binary 0/1 labels.")
        return y

    def _as_1d_array(self, name: str, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array.")
        if len(array) == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite values.")
        return array

    def _to_logits(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(probabilities, 1e-10, 1 - 1e-10)
        return np.log(clipped / (1 - clipped))
