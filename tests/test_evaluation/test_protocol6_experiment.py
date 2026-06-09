import numpy as np
import pandas as pd
import pytest

from experiments.protocol6_subgroup_fairness import main
from src.evaluation import protocol as protocol_module
from src.evaluation.protocol import run_protocol_6


def make_fairness_features():
    rng = np.random.default_rng(42)

    def frame(n, shift):
        rank = np.arange(n)
        x = pd.DataFrame(
            {
                "loan_amnt": 5000 + rank * 100 + shift * 600,
                "dti": 7 + (rank % 50) * 0.5 + shift,
                "fico_avg": 760 - (rank % 80) - shift * 9,
                "revol_util": 12 + (rank % 35) * 1.1 + shift,
            }
        )
        logit = -1.9 + 0.08 * x["dti"] - 0.006 * x["fico_avg"] + 0.00003 * x["loan_amnt"]
        probability = 1 / (1 + np.exp(-logit))
        y = rng.binomial(1, probability)
        if len(np.unique(y)) < 2:
            y[0] = 0
            y[-1] = 1
        return x, y

    X_train, y_train = frame(120, 0.0)
    X_val, y_val = frame(60, 0.3)
    X_test, y_test = frame(80, 0.8)
    groups = pd.DataFrame(
        {
            "state": np.where(np.arange(len(X_test)) % 2 == 0, "CA", "NY"),
            "loan_purpose": np.where(np.arange(len(X_test)) % 4 < 2, "credit_card", "debt_consolidation"),
            "income_group": np.where(np.arange(len(X_test)) % 2 == 0, "income_Q1", "income_Q2"),
        }
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, groups


def make_raw_protocol6_file(tmp_path):
    rows = []
    for year in [2012, 2013, 2014, 2015, 2016, 2017]:
        for index in range(12):
            default = index % 2
            rows.append(
                {
                    "loan_status": "Charged Off" if default else "Fully Paid",
                    "issue_d": f"{year}-0{(index % 9) + 1}",
                    "loan_amnt": 5000 + index * 450 + (year - 2012) * 100,
                    "dti": 8 + index * 1.5 + (year - 2012) * 0.4,
                    "emp_length": index % 10,
                    "addr_state": "CA" if index % 2 else "NY",
                    "purpose": "debt_consolidation" if index % 2 else "credit_card",
                    "zip_code": f"90{index:03d}",
                    "annual_inc": 42000 + index * 3000,
                    "home_ownership": "RENT" if index % 2 else "MORTGAGE",
                    "verification_status": "Verified" if index % 2 else "Not Verified",
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

    data_path = tmp_path / "accepted_fairness.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)
    return data_path


def test_run_protocol_6_reports_subgroup_metrics_and_disparities():
    X_train, y_train, X_val, y_val, X_test, y_test, groups = make_fairness_features()

    result = run_protocol_6(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        groups,
        X_test["loan_amnt"].to_numpy(dtype=float),
        model_names=["LogisticRegression"],
        group_columns=["state", "loan_purpose"],
        approval_pd_threshold=0.5,
        min_group_size=10,
        random_state=7,
    )

    assert result["protocol"].eq("Protocol6").all()
    assert result["model"].eq("LogisticRegression").all()
    assert set(result["group_feature"]) == {"state", "loan_purpose"}
    assert result["skip_reason"].isna().all()
    assert {
        "AUROC",
        "Brier",
        "ECE",
        "approval_rate",
        "bad_rate_gap",
        "equal_opportunity_gap",
        "false_negative_rate_gap",
        "false_positive_rate_gap",
        "manual_review_burden_gap",
        "profit_gap",
        "overall_AUROC",
    }.issubset(result.columns)
    assert result["approval_rate"].between(0, 1).all()
    assert result["manual_review_rate"].between(0, 1).all()
    assert result["n_groups_evaluated"].ge(2).all()


def test_protocol6_main_writes_metrics_csv(tmp_path):
    data_path = make_raw_protocol6_file(tmp_path)
    output_path = tmp_path / "protocol6_metrics.csv"
    cache_path = tmp_path / "accepted_labeled_rich.pkl"

    result = main(
        str(data_path),
        str(output_path),
        model_names=["LogisticRegression"],
        group_columns=["state", "zip3_region", "loan_purpose", "employment_length_group", "income_group"],
        approval_pd_threshold=0.5,
        min_group_size=2,
        cache_path=str(cache_path),
        random_state=7,
    )

    saved = pd.read_csv(output_path)
    assert output_path.exists()
    assert cache_path.exists()
    assert result["test_period"].eq("test_normal").all()
    assert {"state", "zip3_region", "loan_purpose", "employment_length_group", "income_group"}.issubset(
        set(saved["group_feature"])
    )
    assert {"approval_rate_gap", "bad_rate_gap", "equal_opportunity_gap", "profit_gap"}.issubset(saved.columns)


def test_protocol6_forwards_gpu_params_to_supported_baselines(monkeypatch):
    X_train, y_train, X_val, y_val, X_test, y_test, groups = make_fairness_features()
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

    result = run_protocol_6(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        groups,
        X_test["loan_amnt"].to_numpy(dtype=float),
        model_names=["LightGBM"],
        group_columns=["state"],
        approval_pd_threshold=0.5,
        min_group_size=10,
        device_type="gpu",
        gpu_device_id=1,
        random_state=9,
    )

    assert captured == {"random_state": 9, "device_type": "gpu", "gpu_device_id": 1}
    assert result["model"].tolist() == ["LightGBM", "LightGBM"]


def test_run_protocol_6_validates_inputs():
    X_train, y_train, X_val, y_val, X_test, y_test, groups = make_fairness_features()

    with pytest.raises(ValueError, match="same length"):
        run_protocol_6(X_train, y_train[:-1], X_val, y_val, X_test, y_test, groups, X_test["loan_amnt"])

    with pytest.raises(ValueError, match="group_features_test"):
        run_protocol_6(X_train, y_train, X_val, y_val, X_test, y_test, groups.iloc[:-1], X_test["loan_amnt"])

    with pytest.raises(KeyError, match="Unknown Protocol 6 group"):
        run_protocol_6(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            groups,
            X_test["loan_amnt"],
            group_columns=["missing_group"],
        )

    with pytest.raises(ValueError, match="min_group_size"):
        run_protocol_6(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            groups,
            X_test["loan_amnt"],
            min_group_size=0,
        )
