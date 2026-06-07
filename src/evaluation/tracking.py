from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ExperimentTracker:
    """Small MLflow wrapper for reproducible experiment logging."""

    def __init__(self, experiment_name: str, tracking_uri: str | None = None):
        self.mlflow = _import_mlflow()
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment(experiment_name)
        self.run_id: str | None = None

    def start_run(self, run_name: str, config: Mapping[str, Any]) -> str:
        self.mlflow.start_run(run_name=run_name)
        self.mlflow.log_params(flatten_config(config))
        self.run_id = self.mlflow.active_run().info.run_id
        return self.run_id

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        self.mlflow.log_metrics(_coerce_metrics(metrics), step=step)

    def log_artifact(self, path: str | Path) -> None:
        self.mlflow.log_artifact(str(path))

    def log_model(self, model, artifact_path: str) -> None:
        self.mlflow.sklearn.log_model(model, artifact_path)

    def end_run(self) -> None:
        self.mlflow.end_run()

    def log_results_table(self, results: pd.DataFrame, name: str, output_dir: str | Path = ".") -> Path:
        output_path = Path(output_dir) / f"results_{name}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(output_path, index=False)
        self.log_artifact(output_path)
        return output_path


def save_experiment_state(
    config: Mapping[str, Any],
    commit_hash: str,
    data_version: str,
    seed_list: list[int],
    output_dir: str | Path,
) -> Path:
    """Persist the reproducibility state required by the project spec."""
    state = {
        "config": to_jsonable(config),
        "commit_hash": commit_hash,
        "data_version": data_version,
        "seed_list": [int(seed) for seed in seed_list],
    }
    output_path = Path(output_dir) / "experiment_state.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    return output_path


def flatten_config(config: Mapping[str, Any], prefix: str = "") -> dict[str, str | int | float | bool]:
    """Flatten nested config values into MLflow-safe scalar parameters."""
    flat: dict[str, str | int | float | bool] = {}
    for key, value in _mapping_items(config):
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if _is_mapping(value):
            flat.update(flatten_config(value, full_key))
        else:
            flat[full_key] = _to_mlflow_param(value)
    return flat


def to_jsonable(value: Any) -> Any:
    """Convert numpy/pandas/OmegaConf-like values into JSON-serializable objects."""
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    if _is_mapping(value):
        return {str(key): to_jsonable(item) for key, item in _mapping_items(value)}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _import_mlflow():
    try:
        return import_module("mlflow")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mlflow is required for ExperimentTracker. Install project requirements in the active environment."
        ) from exc


def _coerce_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    coerced: dict[str, float] = {}
    for key, value in _mapping_items(metrics):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("metrics must contain finite numeric values.")
        coerced[str(key)] = value
    return coerced


def _to_mlflow_param(value: Any) -> str | int | float | bool:
    value = to_jsonable(value)
    if value is None:
        return "None"
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    return json.dumps(value, sort_keys=True)


def _is_mapping(value: Any) -> bool:
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray, list, tuple, str, bytes, Path)):
        return False
    if isinstance(value, Mapping):
        return True
    if hasattr(value, "items") and not isinstance(value, (str, bytes)):
        return True
    return False


def _mapping_items(value: Mapping[str, Any]):
    return value.items()
