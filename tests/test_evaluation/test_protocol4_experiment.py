import numpy as np
import pandas as pd
import pytest

from experiments.protocol4_temporal_stability import main
from src.evaluation import protocol as protocol_module
from src.evaluation.protocol import run_protocol_4


def make_temporal_features():
    def frame(n, shift):
        rank = np.arange(n)
        x = pd.DataFrame(
            {
                "loan_amnt": 4000 + rank * 120 + shift * 800,
                "dti": 6 + (rank % 40) * 0.6 + shift,
                "fico_avg": 760 - (rank % 70) - shift * 10,
                "revol_util": 15 + (rank % 30) * 1.5 + shift,
            }
        )
        y = (rank % 2).astype(int)
        return x, y

    return {
        "train": frame(90, 0.0),
        "validation": frame(36, 0.2),
        "test_normal": frame(38, 0.5),
        "test_extended": frame(40, 0.9),
        "test_structural_break": frame(42, 1.6),
    }


def make_raw_temporal_lendingclub_file(tmp_path):
    rows = []
    for year in [2012, 2013, 2014, 2015, 2016, 2017, 2018]:
        for index in range(8):
            default = index % 2
            rows.append(
                {
                    "loan_status": "Charged Off" if default else "Fully Paid",
                    "issue_d": f"{year}-0{(index % 9) + 1}",
                    "loan_amnt": 5000 + index * 500 + (year - 2012) * 100,
                    "dti": 8 + index * 2 + (year - 2012),
                    "emp_length": index,
                    "annual_inc": 45000 + index * 3000,
                    "fico_range_low": 650 + index * 5,
                    "fico_range_high": 654 + index * 5,
                    "open_acc": 5 + index,
                    "revol_bal": 1000 + index * 200,
                    "revol_util": 10 + index * 3,
                    "total_acc": 12 + index,
                    "int_rate": 8.0 + index,
                    "installment": 150.0 + index * 10,
                    "delinq_2yrs": index % 3,
                }
            )

    data_path = tmp_path / "accepted_temporal.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)
    return data_path


def test_run_protocol_4_reports_period_stability_metrics():
    data = make_temporal_features()
    X_train, y_train = data["train"]
    periods = {name: value for name, value in data.items() if name != "train"}

    result = run_protocol_4(
        X_train,
        y_train,
        periods,
        model_names=["LogisticRegression"],
        approval_pd_threshold=0.55,
        random_state=7,
    )

    assert result["protocol"].eq("Protocol4").all()
    assert result["model"].tolist() == ["LogisticRegression"] * 4
    assert result["period"].tolist() == ["validation", "test_normal", "test_extended", "test_structural_break"]
    assert result.loc[result["period"].eq("test_structural_break"), "is_structural_break_stress_test"].iloc[0]
    assert {
        "score_psi_vs_train",
        "approval_rate",
        "approved_bad_rate",
        "Brier_drift_vs_validation",
        "ECE_drift_vs_validation",
        "approval_rate_drift_vs_validation",
        "bad_rate_drift_vs_validation",
        "worst_period_AUROC",
        "worst_main_period_AUROC",
    }.issubset(result.columns)
    assert result["score_psi_vs_train"].ge(0).all()
    assert result["approval_rate"].between(0, 1).all()
    assert np.isfinite(result["worst_main_period_AUROC"]).all()


def test_protocol4_main_writes_metrics_csv_and_skips_missing_structural_break(tmp_path):
    data_path = make_raw_temporal_lendingclub_file(tmp_path)
    output_path = tmp_path / "protocol4_metrics.csv"
    cache_path = tmp_path / "accepted_labeled_rich.pkl"

    result = main(
        str(data_path),
        str(output_path),
        model_names=["LogisticRegression"],
        cache_path=str(cache_path),
        random_state=7,
    )

    saved = pd.read_csv(output_path)
    assert output_path.exists()
    assert cache_path.exists()
    assert saved["period"].tolist() == ["validation", "test_normal", "test_extended", "test_structural_break"]
    assert result.loc[result["period"].eq("test_structural_break"), "skip_reason"].iloc[0] == "empty_period"
    assert {"AUROC", "PR-AUC", "Brier", "ECE", "score_psi_vs_train", "approval_rate"}.issubset(saved.columns)
    assert saved.loc[saved["period"].eq("test_normal"), "period_type"].iloc[0] == "normal_drift"
    assert saved.loc[saved["period"].eq("test_extended"), "period_type"].iloc[0] == "extended_drift"


def test_run_protocol_4_validates_inputs():
    data = make_temporal_features()
    X_train, y_train = data["train"]
    periods = {"validation": data["validation"]}

    with pytest.raises(ValueError, match="same length"):
        run_protocol_4(X_train, y_train[:-1], periods, model_names=["LogisticRegression"])

    with pytest.raises(ValueError, match="validation"):
        run_protocol_4(X_train, y_train, {}, model_names=["LogisticRegression"])

    with pytest.raises(ValueError, match="approval_pd_threshold"):
        run_protocol_4(X_train, y_train, periods, approval_pd_threshold=1.5)

    with pytest.raises(KeyError, match="Unknown model"):
        run_protocol_4(X_train, y_train, periods, model_names=["MissingModel"])


def test_protocol4_forwards_gpu_params_to_supported_baselines(monkeypatch):
    data = make_temporal_features()
    X_train, y_train = data["train"]
    periods = {"validation": data["validation"]}
    captured = {}

    class DummyProbabilisticModel:
        def __init__(self, random_state=42, device_type="cpu", gpu_device_id=0):
            captured["random_state"] = random_state
            captured["device_type"] = device_type
            captured["gpu_device_id"] = gpu_device_id

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            probabilities = np.linspace(0.2, 0.8, len(X))
            return np.column_stack([1.0 - probabilities, probabilities])

    monkeypatch.setitem(protocol_module.TRADITIONAL_BASELINES, "LightGBM", DummyProbabilisticModel)

    result = run_protocol_4(
        X_train,
        y_train,
        periods,
        model_names=["LightGBM"],
        device_type="gpu",
        gpu_device_id=1,
        random_state=9,
    )

    assert captured == {"random_state": 9, "device_type": "gpu", "gpu_device_id": 1}
    assert result["model"].tolist() == ["LightGBM"]
