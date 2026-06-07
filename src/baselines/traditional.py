from __future__ import annotations

import warnings
from importlib.util import find_spec

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def _build_gradient_boosting_fallback(random_state: int = 42) -> BaseEstimator:
    return GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=random_state,
    )


def build_logistic_regression(random_state: int = 42) -> BaseEstimator:
    return LogisticRegression(C=1.0, max_iter=2000, random_state=random_state, solver="liblinear")


def build_random_forest(random_state: int = 42) -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )


def build_xgboost(random_state: int = 42) -> BaseEstimator:
    if find_spec("xgboost") is None:
        return _build_gradient_boosting_fallback(random_state)

    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=1,
        tree_method="hist",
    )


def build_lightgbm(random_state: int = 42) -> BaseEstimator:
    if find_spec("lightgbm") is None:
        return _build_gradient_boosting_fallback(random_state)

    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )


def build_catboost(random_state: int = 42) -> BaseEstimator:
    if find_spec("catboost") is None:
        return _build_gradient_boosting_fallback(random_state)

    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=100,
        depth=5,
        random_seed=random_state,
        silent=True,
        allow_writing_files=False,
    )


def build_mlp(random_state: int = 42) -> BaseEstimator:
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        random_state=random_state,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
    )


def build_mlp_focal(random_state: int = 42) -> BaseEstimator:
    """Proxy for the supplementary MLP-focal baseline in the project spec."""
    return build_mlp(random_state)


def build_smote_baseline(random_state: int = 42, use_imblearn: bool = False):
    """SMOTE plus LightGBM when explicitly enabled; otherwise a stable LightGBM fallback."""
    if not use_imblearn or find_spec("imblearn") is None:
        return build_lightgbm(random_state)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline
    except Exception:
        return build_lightgbm(random_state)

    return Pipeline(
        steps=[
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", build_lightgbm(random_state)),
        ]
    )


def build_ft_transformer(random_state: int = 42):
    """Sklearn proxy for the optional FT-Transformer tabular baseline."""
    return MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        random_state=random_state,
        max_iter=300,
        early_stopping=True,
    )


def build_tabnet(random_state: int = 42):
    """TabNet baseline when pytorch-tabnet is installed; otherwise an MLP proxy."""
    if find_spec("pytorch_tabnet") is None:
        return build_mlp(random_state)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            from pytorch_tabnet.tab_model import TabNetClassifier
    except Exception:
        return build_mlp(random_state)

    return TabNetClassifier(seed=random_state, verbose=0)


def build_saint(random_state: int = 42):
    """MLP proxy for the optional SAINT tabular baseline."""
    return build_ft_transformer(random_state)


TRADITIONAL_BASELINES = {
    "LogisticRegression": build_logistic_regression,
    "RandomForest": build_random_forest,
    "XGBoost": build_xgboost,
    "LightGBM": build_lightgbm,
    "CatBoost": build_catboost,
    "MLP": build_mlp,
    "FT-Transformer": build_ft_transformer,
    "TabNet": build_tabnet,
    "SAINT": build_saint,
    "MLP-Focal": build_mlp_focal,
    "LightGBM-SMOTE": build_smote_baseline,
}

SUPPLEMENTARY_BASELINES = {"MLP-Focal", "LightGBM-SMOTE"}
DEEP_TABULAR_BASELINES = {"FT-Transformer", "TabNet", "SAINT"}
