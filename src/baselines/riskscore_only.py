from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.data.risk_score import RISK_SCORE_COLUMN, find_riskscore_column


class RiskScoreBinning:
    """Risk_Score scorecard-style binning baseline."""

    def __init__(self, n_bins: int = 20):
        if n_bins <= 0:
            raise ValueError("n_bins must be positive.")
        self.n_bins = int(n_bins)
        self.bin_edges_: np.ndarray | None = None
        self.bin_rates_: dict[int, float] = {}
        self.global_rate_: float | None = None
        self.score_median_: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> RiskScoreBinning:
        scores = _risk_score_array(X)
        labels = _validate_binary_labels(y, expected_length=len(scores))
        self.global_rate_ = float(labels.mean())
        self.score_median_ = float(np.nanmedian(scores))
        scores = _fill_missing_scores(scores, self.score_median_)

        unique_scores = np.unique(scores)
        if len(unique_scores) <= 1:
            self.bin_edges_ = np.array([-np.inf, np.inf], dtype=float)
            self.bin_rates_ = {0: self.global_rate_}
            return self

        n_bins = min(self.n_bins, len(unique_scores))
        bin_codes, edges = pd.qcut(scores, q=n_bins, labels=False, retbins=True, duplicates="drop")
        edges = np.asarray(edges, dtype=float)
        edges[0] = -np.inf
        edges[-1] = np.inf
        self.bin_edges_ = edges
        self.bin_rates_ = {}
        codes = np.asarray(bin_codes, dtype=int)
        for code in np.unique(codes):
            self.bin_rates_[int(code)] = float(labels[codes == code].mean())
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.bin_edges_ is not None
        assert self.global_rate_ is not None
        assert self.score_median_ is not None

        scores = _fill_missing_scores(_risk_score_array(X), self.score_median_)
        codes = np.digitize(scores, self.bin_edges_[1:-1], right=True)
        probabilities = np.array([self.bin_rates_.get(int(code), self.global_rate_) for code in codes], dtype=float)
        probabilities = np.clip(probabilities, 0.0, 1.0)
        return np.column_stack([1.0 - probabilities, probabilities])

    def _check_fitted(self) -> None:
        if self.bin_edges_ is None or self.global_rate_ is None:
            raise ValueError("RiskScoreBinning must be fitted before prediction.")


class RiskScoreLogisticRegression:
    """Logistic regression baseline using Risk_Score only."""

    def __init__(self, random_state: int = 42, C: float = 1.0):
        self.random_state = random_state
        self.C = C
        self.model = LogisticRegression(C=C, max_iter=2000, solver="liblinear", random_state=random_state)
        self.features_: list[str] | None = None
        self.medians_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RiskScoreLogisticRegression:
        matrix = _feature_matrix(X, [RISK_SCORE_COLUMN])
        labels = _validate_binary_labels(y, expected_length=len(matrix))
        self.features_ = [RISK_SCORE_COLUMN]
        self.medians_ = matrix.median(numeric_only=True).fillna(0.0)
        self.model.fit(matrix.fillna(self.medians_), labels)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.features_ is None or self.medians_ is None:
            raise ValueError("RiskScoreLogisticRegression must be fitted before prediction.")
        matrix = _feature_matrix(X, self.features_).fillna(self.medians_)
        return self.model.predict_proba(matrix)


class RiskScoreDtiLogisticRegression(RiskScoreLogisticRegression):
    """Simple Risk_Score + DTI logistic baseline."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RiskScoreDtiLogisticRegression:
        available_features = [RISK_SCORE_COLUMN]
        if "dti" in pd.DataFrame(X).columns:
            available_features.append("dti")
        matrix = _feature_matrix(X, available_features)
        labels = _validate_binary_labels(y, expected_length=len(matrix))
        self.features_ = available_features
        self.medians_ = matrix.median(numeric_only=True).fillna(0.0)
        self.model.fit(matrix.fillna(self.medians_), labels)
        return self


class RiskScoreIsotonicRegression:
    """Monotone isotonic calibration baseline using Risk_Score only."""

    def __init__(self, out_of_bounds: str = "clip"):
        self.out_of_bounds = out_of_bounds
        self.model: IsotonicRegression | None = None
        self.score_median_: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> RiskScoreIsotonicRegression:
        scores = _risk_score_array(X)
        labels = _validate_binary_labels(y, expected_length=len(scores))
        self.score_median_ = float(np.nanmedian(scores))
        scores = _fill_missing_scores(scores, self.score_median_)
        increasing = _infer_increasing_direction(scores, labels)
        self.model = IsotonicRegression(increasing=increasing, out_of_bounds=self.out_of_bounds)
        self.model.fit(scores, labels.astype(float))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model is None or self.score_median_ is None:
            raise ValueError("RiskScoreIsotonicRegression must be fitted before prediction.")
        scores = _fill_missing_scores(_risk_score_array(X), self.score_median_)
        probabilities = np.clip(np.asarray(self.model.predict(scores), dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - probabilities, probabilities])


def build_riskscore_only_models(random_state: int = 42) -> dict[str, object]:
    return {
        "risk_score_binning": RiskScoreBinning(),
        "risk_score_lr": RiskScoreLogisticRegression(random_state=random_state),
        "risk_score_dti_lr": RiskScoreDtiLogisticRegression(random_state=random_state),
        "risk_score_isotonic": RiskScoreIsotonicRegression(),
    }


def build_riskscore_binning(random_state: int = 42) -> RiskScoreBinning:
    _ = random_state
    return RiskScoreBinning()


def build_riskscore_lr(random_state: int = 42) -> RiskScoreLogisticRegression:
    return RiskScoreLogisticRegression(random_state=random_state)


def build_riskscore_dti_lr(random_state: int = 42) -> RiskScoreDtiLogisticRegression:
    return RiskScoreDtiLogisticRegression(random_state=random_state)


def build_riskscore_isotonic(random_state: int = 42) -> RiskScoreIsotonicRegression:
    _ = random_state
    return RiskScoreIsotonicRegression()


def fit_riskscore_lr(X: pd.DataFrame, y: np.ndarray, random_state: int = 42) -> RiskScoreLogisticRegression:
    return build_riskscore_lr(random_state=random_state).fit(X, y)


def fit_riskscore_dti_lr(X: pd.DataFrame, y: np.ndarray, random_state: int = 42) -> RiskScoreDtiLogisticRegression:
    return build_riskscore_dti_lr(random_state=random_state).fit(X, y)


RISKSCORE_ONLY_BASELINES = {
    "risk_score_binning": build_riskscore_binning,
    "risk_score_lr": build_riskscore_lr,
    "risk_score_dti_lr": build_riskscore_dti_lr,
    "risk_score_isotonic": build_riskscore_isotonic,
}


def _risk_score_array(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        score_column = find_riskscore_column(X)
        if score_column is None:
            raise KeyError("Risk_Score baseline requires a risk_score column.")
        values = X[score_column]
    else:
        array = np.asarray(X)
        if array.ndim == 1:
            values = array
        elif array.ndim == 2 and array.shape[1] >= 1:
            values = array[:, 0]
        else:
            raise ValueError("Risk_Score array must be one-dimensional or a two-dimensional matrix.")
    scores = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if len(scores) == 0:
        raise ValueError("Risk_Score values must not be empty.")
    if np.all(np.isnan(scores)):
        raise ValueError("Risk_Score values must contain at least one numeric value.")
    return scores


def _feature_matrix(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(X).copy()
    score_column = find_riskscore_column(frame)
    if score_column is None:
        raise KeyError("Risk_Score baseline requires a risk_score column.")
    frame[RISK_SCORE_COLUMN] = frame[score_column]

    missing = [feature for feature in feature_names if feature not in frame.columns]
    if missing:
        raise KeyError(f"Missing required feature(s): {missing}")
    matrix = frame[feature_names].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if matrix.empty:
        raise ValueError("Risk_Score baseline requires at least one feature.")
    return matrix


def _validate_binary_labels(y: np.ndarray, expected_length: int) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")
    if len(labels) != expected_length:
        raise ValueError("X and y must have the same length.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y must contain binary 0/1 labels.")
    return labels.astype(int)


def _fill_missing_scores(scores: np.ndarray, median: float) -> np.ndarray:
    return np.nan_to_num(np.asarray(scores, dtype=float), nan=median, posinf=median, neginf=median)


def _infer_increasing_direction(scores: np.ndarray, y: np.ndarray) -> bool:
    if len(np.unique(scores)) <= 1:
        return True
    centered_scores = scores.astype(float) - float(np.mean(scores))
    centered_labels = y.astype(float) - float(np.mean(y))
    denominator = float(np.sqrt(np.sum(centered_scores**2) * np.sum(centered_labels**2)))
    if denominator <= 0:
        return True
    correlation = float(np.sum(centered_scores * centered_labels) / denominator)
    return bool(correlation >= 0)
