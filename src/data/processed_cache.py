from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import load_accepted
from src.data.preprocess import build_accepted_rich_features, construct_default_label, label_maturity_filter


DEFAULT_ACCEPTED_RICH_CACHE_PATH = Path("data/processed/lendingclub/accepted_labeled_rich.pkl")
REQUIRED_ACCEPTED_RICH_COLUMNS = {"application_date", "default_label", "issue_d"}


def build_accepted_labeled_rich_frame(data_path: str | Path) -> pd.DataFrame:
    """Build the labeled accepted-rich modeling frame shared by accepted-only protocols."""
    accepted = load_accepted(data_path)
    labeled = construct_default_label(label_maturity_filter(accepted)).dropna(subset=["default_label"]).copy()

    modeling_frame = build_accepted_rich_features(labeled)
    if "issue_d" not in modeling_frame.columns:
        raise KeyError("Accepted data must contain issue_d for time-based protocol splits.")

    modeling_frame["application_date"] = modeling_frame["issue_d"]
    modeling_frame["default_label"] = labeled["default_label"].astype(int).to_numpy()
    return modeling_frame.reset_index(drop=True)


def load_or_build_accepted_labeled_rich(
    data_path: str | Path,
    cache_path: str | Path | None = DEFAULT_ACCEPTED_RICH_CACHE_PATH,
    refresh_cache: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Load the accepted-rich modeling frame from cache, or build and cache it from raw CSV.

    Returns the frame plus a boolean cache-hit flag.
    """
    resolved_cache_path = Path(cache_path) if cache_path is not None else None
    if resolved_cache_path is not None and resolved_cache_path.exists() and not refresh_cache:
        cached = pd.read_pickle(resolved_cache_path)
        _validate_accepted_rich_cache(cached, resolved_cache_path)
        return cached.reset_index(drop=True), True

    frame = build_accepted_labeled_rich_frame(data_path)
    if resolved_cache_path is not None:
        resolved_cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(resolved_cache_path)
    return frame, False


def _validate_accepted_rich_cache(frame: pd.DataFrame, cache_path: Path) -> None:
    missing = REQUIRED_ACCEPTED_RICH_COLUMNS.difference(frame.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Accepted-rich cache {cache_path} is missing required columns: {missing_columns}")
