from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalStringifier(BaseEstimator, TransformerMixin):
    """Convert categorical feature values to strings before one-hot encoding."""

    def __init__(self, missing_value: str = "missing"):
        self.missing_value = missing_value

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def transform(self, X) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        if hasattr(self, "feature_names_in_") and len(frame.columns) == len(self.feature_names_in_):
            frame.columns = self.feature_names_in_
        return frame.astype("string").fillna(self.missing_value).astype(str)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_
