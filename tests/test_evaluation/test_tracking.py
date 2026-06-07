import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.evaluation.tracking import ExperimentTracker, flatten_config, save_experiment_state, to_jsonable


class FakeMLflow:
    def __init__(self):
        self.tracking_uri = None
        self.experiment_name = None
        self.started_run_name = None
        self.params = None
        self.metrics = []
        self.artifacts = []
        self.logged_model = None
        self.ended = False
        self.sklearn = SimpleNamespace(log_model=self.log_model)

    def set_tracking_uri(self, tracking_uri):
        self.tracking_uri = tracking_uri

    def set_experiment(self, experiment_name):
        self.experiment_name = experiment_name

    def start_run(self, run_name):
        self.started_run_name = run_name

    def log_params(self, params):
        self.params = params

    def active_run(self):
        return SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

    def log_metrics(self, metrics, step=None):
        self.metrics.append((metrics, step))

    def log_artifact(self, path):
        self.artifacts.append(path)

    def log_model(self, model, artifact_path):
        self.logged_model = (model, artifact_path)

    def end_run(self):
        self.ended = True


def test_experiment_tracker_logs_run_params_metrics_artifacts_and_model(monkeypatch, tmp_path):
    fake_mlflow = FakeMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    tracker = ExperimentTracker("ucri-cs-test", tracking_uri=str(tmp_path / "mlruns"))

    run_id = tracker.start_run(
        "smoke",
        {
            "experiment": {"seed": 42},
            "teacher": {"model_types": ["lightgbm", "catboost"]},
        },
    )
    tracker.log_metrics({"AUROC": np.float64(0.7), "Brier": 0.2}, step=3)
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("ok", encoding="utf-8")
    tracker.log_artifact(artifact)
    model = object()
    tracker.log_model(model, "student")
    tracker.end_run()

    assert fake_mlflow.tracking_uri == str(tmp_path / "mlruns")
    assert fake_mlflow.experiment_name == "ucri-cs-test"
    assert fake_mlflow.started_run_name == "smoke"
    assert run_id == "run-123"
    assert fake_mlflow.params["experiment.seed"] == 42
    assert fake_mlflow.params["teacher.model_types"] == '["lightgbm", "catboost"]'
    assert fake_mlflow.metrics == [({"AUROC": 0.7, "Brier": 0.2}, 3)]
    assert fake_mlflow.artifacts == [str(artifact)]
    assert fake_mlflow.logged_model == (model, "student")
    assert fake_mlflow.ended


def test_log_results_table_writes_csv_and_logs_artifact(monkeypatch, tmp_path):
    fake_mlflow = FakeMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    tracker = ExperimentTracker("ucri-cs-test")
    results = pd.DataFrame({"model": ["accepted-only"], "AUROC": [0.69]})

    output_path = tracker.log_results_table(results, "leaderboard", output_dir=tmp_path)

    assert output_path == tmp_path / "results_leaderboard.csv"
    assert output_path.exists()
    assert pd.read_csv(output_path).to_dict(orient="records") == [{"model": "accepted-only", "AUROC": 0.69}]
    assert fake_mlflow.artifacts == [str(output_path)]


def test_save_experiment_state_writes_reproducibility_json(tmp_path):
    output_path = save_experiment_state(
        config={
            "evaluation": {"seeds": np.array([42, 43])},
            "paths": {"root": Path("data/raw")},
        },
        commit_hash="abc123",
        data_version="lendingclub-sha256",
        seed_list=[42, 43],
        output_dir=tmp_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["commit_hash"] == "abc123"
    assert payload["data_version"] == "lendingclub-sha256"
    assert payload["seed_list"] == [42, 43]
    assert payload["config"]["evaluation"]["seeds"] == [42, 43]
    assert payload["config"]["paths"]["root"] == "data/raw"


def test_flatten_config_and_to_jsonable_handle_common_experiment_values():
    config = {
        "teacher": {"alpha": np.array([0.25, 0.75])},
        "student": {"post_calibrate": True},
        "table": pd.DataFrame({"metric": ["AUROC"], "value": [0.7]}),
    }

    flat = flatten_config(config)

    assert flat["teacher.alpha"] == "[0.25, 0.75]"
    assert flat["student.post_calibrate"] is True
    assert flat["table"] == '[{"metric": "AUROC", "value": 0.7}]'
    assert to_jsonable(pd.Series([np.int64(1), np.int64(2)])) == [1, 2]


def test_log_metrics_rejects_non_finite_values(monkeypatch):
    fake_mlflow = FakeMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    tracker = ExperimentTracker("ucri-cs-test")

    with pytest.raises(ValueError, match="finite"):
        tracker.log_metrics({"AUROC": np.nan})
