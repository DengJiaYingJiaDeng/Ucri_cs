from __future__ import annotations

import pandas as pd

from src.data.loader import FORBIDDEN_FEATURES


class ForbiddenFeatureError(ValueError):
    """Raised when post-approval or label-proxy fields enter a feature matrix."""


def audit_features(df: pd.DataFrame) -> None:
    """Raise if a dataframe contains forbidden training features."""
    forbidden_found = [column for column in FORBIDDEN_FEATURES if column in df.columns]
    if forbidden_found:
        raise ForbiddenFeatureError(
            f"Forbidden features in dataframe: {forbidden_found}. "
            "These features contain post-approval information and cannot be used for training."
        )
