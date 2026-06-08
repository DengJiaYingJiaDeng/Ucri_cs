from __future__ import annotations

import warnings


LIGHTGBM_FEATURE_NAME_WARNING = (
    r"X does not have valid feature names, but LGBMClassifier was fitted with feature names"
)


def predict_proba_silencing_lightgbm_feature_name_warning(model, X):
    """Run predict_proba while silencing LightGBM's Pipeline-internal feature-name warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=LIGHTGBM_FEATURE_NAME_WARNING,
            category=UserWarning,
            module=r"sklearn\.utils\.validation",
        )
        return model.predict_proba(X)
