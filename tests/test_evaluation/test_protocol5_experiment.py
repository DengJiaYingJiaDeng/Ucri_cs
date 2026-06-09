import numpy as np
import pandas as pd
import pytest

from experiments.protocol5_decision_approval import main
from src.evaluation import protocol as protocol_module
from src.evaluation.protocol import run_protocol_5


def make_decision_features():
    rng = np.random.default_rng(42)

    def frame(n, shift):
        rank = np.arange(n)
        x = pd.DataFrame(
            {
                "loan_amnt": 3000 + rank * 120 + shift * 700,
                "dti": 5 + (rank % 50) * 0.6 + shift,
                "fico_avg": 760 - (rank % 70) - shift * 12,
                "revol_util": 12 + (rank % 30) * 1.2 + shift,
            }
        )
        logit = -1.8 + 0.09 * x["dti"] - 0.007 * x["fico_avg"] + 0.00002 * x["loan_amnt"]
        probability = 1 / (1 + np.exp(-logit))
        y = rng.binomial(1, probability)
        if len(np.unique(y)) < 2:
            y[0] = 0
            y[-1] = 1
        return x, y

    return (*frame(100, 0.0), *frame(50, 0.4), *frame(60, 0.8))


def make_raw_protocol5_file(tmp_path):
    rows = []
    for year in [2012, 2013, 2014, 2015, 2016, 2017]:
        for index in range(10):
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

    data_path = tmp_path / "accepted_decision.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)
    return data_path


def test_run_protocol_5_returns_profit_risk_frontier_rows():
    X_train, y_train, X_val, y_val, X_test, y_test = make_decision_features()
    loan_amounts = X_test["loan_amnt"].to_numpy(dtype=float)

    result = run_protocol_5(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        loan_amounts,
        model_names=["LogisticRegression"],
        target_bad_rates=[0.08],
        min_approval_rates=[0.2, 0.4],
        lgd_values=[0.2, 0.45],
        random_state=7,
    )

    assert len(result) == 4
    assert result["protocol"].eq("Protocol5").all()
    assert result["model"].eq("LogisticRegression").all()
    assert {
        "expected_profit",
        "approval_rate",
        "realized_bad_rate",
        "average_calibrated_pd",
        "ks_at_approval_boundary",
        "manual_review_rate",
        "validation_constraint_feasible",
        "oracle_profit",
        "random_profit",
        "oracle_profit_ratio",
    }.issubset(result.columns)
    assert result["approval_rate"].between(0, 1).all()
    assert result["manual_review_rate"].between(0, 1).all()
    assert result["lgd"].tolist() == [0.2, 0.45, 0.2, 0.45]


def test_protocol5_main_writes_metrics_csv(tmp_path):
    data_path = make_raw_protocol5_file(tmp_path)
    output_path = tmp_path / "protocol5_metrics.csv"
    cache_path = tmp_path / "accepted_labeled_rich.pkl"

    result = main(
        str(data_path),
        str(output_path),
        model_names=["LogisticRegression"],
        target_bad_rates=[0.08],
        min_approval_rates=[0.2],
        lgd_values=[0.45],
        cache_path=str(cache_path),
        random_state=7,
    )

    saved = pd.read_csv(output_path)
    assert output_path.exists()
    assert cache_path.exists()
    assert result["test_period"].eq("test_normal").all()
    assert saved["protocol"].tolist() == ["Protocol5"]
    assert {"expected_profit", "profit_per_loan", "approval_rate", "realized_bad_rate"}.issubset(saved.columns)


def test_protocol5_forwards_gpu_params_to_supported_baselines(monkeypatch):
    X_train, y_train, X_val, y_val, X_test, y_test = make_decision_features()
    captured = {}

    class DummyProbabilisticModel:
        def __init__(self, random_state=42, device_type="cpu", gpu_device_id=0):
            captured["random_state"] = random_state
            captured["device_type"] = device_type
            captured["gpu_device_id"] = gpu_device_id

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            probabilities = np.linspace(0.1, 0.9, len(X))
            return np.column_stack([1.0 - probabilities, probabilities])

    monkeypatch.setitem(protocol_module.TRADITIONAL_BASELINES, "LightGBM", DummyProbabilisticModel)

    run_protocol_5(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_test["loan_amnt"].to_numpy(dtype=float),
        model_names=["LightGBM"],
        target_bad_rates=[0.08],
        min_approval_rates=[0.2],
        lgd_values=[0.45],
        device_type="gpu",
        gpu_device_id=1,
        random_state=9,
    )

    assert captured == {"random_state": 9, "device_type": "gpu", "gpu_device_id": 1}


def test_run_protocol_5_validates_inputs():
    X_train, y_train, X_val, y_val, X_test, y_test = make_decision_features()

    with pytest.raises(ValueError, match="same length"):
        run_protocol_5(X_train, y_train[:-1], X_val, y_val, X_test, y_test, X_test["loan_amnt"])

    with pytest.raises(ValueError, match="loan_amounts_test"):
        run_protocol_5(X_train, y_train, X_val, y_val, X_test, y_test, X_test["loan_amnt"].iloc[:-1])

    with pytest.raises(ValueError, match="target_bad_rates"):
        run_protocol_5(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            X_test["loan_amnt"],
            target_bad_rates=[1.2],
        )
