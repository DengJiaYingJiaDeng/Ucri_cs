from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.models.student import StudentModel
from src.models.teacher import TeacherEnsemble
from src.reject_inference.pseudo_label import PseudoLabeler
from src.uncertainty.composite import CompositeUncertainty


class UCRITrainer:
    """Semi-supervised UCRI-CS training loop."""

    def __init__(
        self,
        teacher_config: dict | None = None,
        student_model_type: str = "lightgbm",
        tau_u: float = 0.5,
        gamma: float = 2.0,
        lambda_distill: float = 0.3,
        random_state: int = 42,
        device_type: str = "cpu",
        gpu_device_id: int = 0,
    ):
        self.teacher_config = dict(teacher_config or {"n_models": 5})
        self.student_model_type = student_model_type
        self.tau_u = tau_u
        self.gamma = gamma
        self.lambda_distill = self._validate_non_negative("lambda_distill", lambda_distill)
        self.random_state = random_state
        self.device_type = device_type
        self.gpu_device_id = gpu_device_id

        self.teacher = self._make_teacher()
        self.student = self._make_student()
        self.composite_unc = CompositeUncertainty()
        self.pseudo_labeler = PseudoLabeler(tau_u=tau_u, gamma=gamma)

    def run(
        self,
        X_labeled: pd.DataFrame,
        y_labeled: np.ndarray,
        X_unlabeled: pd.DataFrame,
        X_calib: pd.DataFrame | None = None,
        y_calib: np.ndarray | None = None,
    ) -> dict[str, object]:
        x_labeled, y_labeled, x_unlabeled = self._validate_training_inputs(X_labeled, y_labeled, X_unlabeled)
        x_calib, y_calib = self._validate_optional_calibration(X_calib, y_calib)

        self.teacher = self._make_teacher()
        self.student = self._make_student()
        self.composite_unc = CompositeUncertainty()
        self.pseudo_labeler = PseudoLabeler(tau_u=self.tau_u, gamma=self.gamma)

        self.teacher.fit(x_labeled, y_labeled)
        self._calibrate_teacher_if_requested(self.teacher, x_calib, y_calib)

        teacher_probs = self._predict_teacher_probabilities(self.teacher, x_unlabeled)
        uncertainty = self.composite_unc.compute_from_teacher(x_unlabeled, x_labeled, self.teacher)
        pseudo_result = self.pseudo_labeler.label(x_unlabeled, teacher_probs, uncertainty)

        self.student.fit(
            x_labeled,
            y_labeled,
            x_unlabeled,
            pseudo_result["soft_label"],
            pseudo_result["weight"],
            lambda_distill=self.lambda_distill,
        )

        return {
            "teacher": self.teacher,
            "student": self.student,
            "pseudo_labels": pseudo_result,
            "uncertainty": uncertainty,
            "teacher_probs": teacher_probs,
        }

    def run_out_of_fold(
        self,
        X_labeled: pd.DataFrame,
        y_labeled: np.ndarray,
        X_unlabeled: pd.DataFrame,
        n_folds: int = 5,
        X_calib: pd.DataFrame | None = None,
        y_calib: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Generate accepted OOF teacher probabilities and fold-averaged rejected pseudo-labels."""
        x_labeled, y_labeled, x_unlabeled = self._validate_training_inputs(X_labeled, y_labeled, X_unlabeled)
        x_calib, y_calib = self._validate_optional_calibration(X_calib, y_calib)
        self._validate_n_folds(y_labeled, n_folds)

        oof_probs = np.zeros(len(x_labeled), dtype=float)
        unlabeled_fold_probs = np.zeros((n_folds, len(x_unlabeled)), dtype=float)

        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        for fold_index, (train_index, holdout_index) in enumerate(splitter.split(x_labeled, y_labeled)):
            fold_teacher = self._make_teacher(seed_offset=fold_index + 1)
            fold_teacher.fit(x_labeled.iloc[train_index], y_labeled[train_index])
            self._calibrate_teacher_if_requested(fold_teacher, x_calib, y_calib)

            oof_probs[holdout_index] = self._predict_teacher_probabilities(
                fold_teacher,
                x_labeled.iloc[holdout_index],
            )
            unlabeled_fold_probs[fold_index] = self._predict_teacher_probabilities(fold_teacher, x_unlabeled)

        self.teacher = self._make_teacher()
        self.student = self._make_student()
        self.composite_unc = CompositeUncertainty()
        self.pseudo_labeler = PseudoLabeler(tau_u=self.tau_u, gamma=self.gamma)

        self.teacher.fit(x_labeled, y_labeled)
        self._calibrate_teacher_if_requested(self.teacher, x_calib, y_calib)

        teacher_probs = unlabeled_fold_probs.mean(axis=0)
        uncertainty = self.composite_unc.compute_from_teacher(x_unlabeled, x_labeled, self.teacher)
        pseudo_result = self.pseudo_labeler.label(x_unlabeled, teacher_probs, uncertainty)

        self.student.fit(
            x_labeled,
            y_labeled,
            x_unlabeled,
            pseudo_result["soft_label"],
            pseudo_result["weight"],
            lambda_distill=self.lambda_distill,
        )

        return {
            "teacher": self.teacher,
            "student": self.student,
            "pseudo_labels": pseudo_result,
            "uncertainty": uncertainty,
            "teacher_probs": teacher_probs,
            "oof_probs": oof_probs,
        }

    def _make_teacher(self, seed_offset: int = 0) -> TeacherEnsemble:
        config = dict(self.teacher_config)
        return TeacherEnsemble(
            n_models=config.get("n_models", 5),
            model_types=config.get("model_types"),
            random_state=config.get("random_state", self.random_state + seed_offset),
            class_weight=config.get("class_weight", "balanced"),
            scale_pos_weight=config.get("scale_pos_weight"),
            device_type=config.get("device_type", self.device_type),
            gpu_device_id=config.get("gpu_device_id", self.gpu_device_id),
        )

    def _make_student(self) -> StudentModel:
        return StudentModel(
            model_type=self.student_model_type,
            random_state=self.random_state,
            device_type=self.device_type,
            gpu_device_id=self.gpu_device_id,
        )

    def _predict_teacher_probabilities(self, teacher: TeacherEnsemble, x: pd.DataFrame) -> np.ndarray:
        probabilities = teacher.predict_calibrated(x) if teacher.calibrated else teacher.predict_proba(x)
        return np.clip(probabilities, 0.0, 1.0)

    def _calibrate_teacher_if_requested(
        self,
        teacher: TeacherEnsemble,
        x_calib: pd.DataFrame | None,
        y_calib: np.ndarray | None,
    ) -> None:
        if x_calib is not None and y_calib is not None:
            teacher.calibrate(x_calib, y_calib)

    def _validate_training_inputs(
        self,
        X_labeled: pd.DataFrame,
        y_labeled: np.ndarray,
        X_unlabeled: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        x_labeled = pd.DataFrame(X_labeled).copy()
        x_unlabeled = pd.DataFrame(X_unlabeled).copy().reindex(columns=x_labeled.columns)
        y_labeled = self._validate_binary_labels("y_labeled", y_labeled, expected_length=len(x_labeled))
        if len(x_unlabeled) == 0:
            raise ValueError("X_unlabeled must not be empty.")
        return x_labeled, y_labeled, x_unlabeled

    def _validate_optional_calibration(
        self,
        X_calib: pd.DataFrame | None,
        y_calib: np.ndarray | None,
    ) -> tuple[pd.DataFrame | None, np.ndarray | None]:
        if X_calib is None and y_calib is None:
            return None, None
        if X_calib is None or y_calib is None:
            raise ValueError("X_calib and y_calib must be provided together.")

        x_calib = pd.DataFrame(X_calib).copy()
        y_calib = self._validate_binary_labels("y_calib", y_calib, expected_length=len(x_calib))
        return x_calib, y_calib

    def _validate_binary_labels(self, name: str, labels: np.ndarray, expected_length: int) -> np.ndarray:
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array.")
        if len(labels) != expected_length:
            raise ValueError("X_labeled and y_labeled must have the same length.")
        if len(labels) == 0:
            raise ValueError(f"{name} must not be empty.")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{name} must contain binary 0/1 labels.")
        return labels.astype(int)

    def _validate_n_folds(self, y: np.ndarray, n_folds: int) -> None:
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2.")
        class_counts = np.bincount(y, minlength=2)
        if np.any(class_counts < n_folds):
            raise ValueError("n_folds must not exceed the count of either class.")

    def _validate_non_negative(self, name: str, value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a non-negative finite value.")
        return value
