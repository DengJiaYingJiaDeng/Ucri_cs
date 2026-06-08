import numpy as np
import pandas as pd
import pytest

from experiments.protocol2_real_rejected_ssl import main, run_protocol_2


def make_protocol2_features(n_train=120, n_test=60, n_rejected=45):
    rng = np.random.default_rng(42)

    def make_frame(n, shift=0.0):
        rank = np.arange(n)
        return pd.DataFrame(
            {
                "loan_amount": 5000 + rank * 120 + shift * 1000,
                "dti": 6 + (rank % 50) * 0.7 + shift,
                "emp_length": rank % 20,
                "fico_avg": 760 - (rank % 80) - shift * 15,
                "state": rng.choice(["CA", "NY", "TX"], n),
                "loan_purpose": rng.choice(["debt_consolidation", "credit_card", "medical"], n),
            }
        )

    x_train = make_frame(n_train, shift=0.0)
    x_test = make_frame(n_test, shift=0.2)
    x_rejected = make_frame(n_rejected, shift=1.0)
    y_train = (np.arange(n_train) % 2).astype(int)
    y_test = (np.arange(n_test) % 2).astype(int)
    return x_train, y_train, x_test, y_test, x_rejected


def make_raw_lendingclub_files(tmp_path):
    accepted_rows = []
    rejected_rows = []
    for year in [2012, 2013, 2014, 2016, 2017]:
        for index in range(8):
            bad = index % 2
            accepted_rows.append(
                {
                    "loan_status": "Charged Off" if bad else "Fully Paid",
                    "issue_d": f"{year}-0{(index % 9) + 1}",
                    "loan_amnt": 5000 + index * 400 + (year - 2012) * 50,
                    "dti": 8 + index * 2 + (year - 2012) * 0.2,
                    "emp_length": index,
                    "addr_state": "CA" if index % 2 else "NY",
                    "purpose": "debt_consolidation" if index % 2 else "credit_card",
                    "home_ownership": "RENT" if index % 2 else "MORTGAGE",
                    "annual_inc": 45000 + index * 2500,
                    "verification_status": "Verified" if index % 2 else "Not Verified",
                    "delinq_2yrs": index % 3,
                    "open_acc": 5 + index,
                    "revol_bal": 1000 + index * 200,
                    "revol_util": 10 + index * 3,
                    "total_acc": 12 + index,
                    "fico_range_low": 650 + index * 6,
                    "fico_range_high": 654 + index * 6,
                    "zip_code": f"90{index:03d}",
                }
            )

    for year in [2012, 2013, 2014]:
        for index in range(7):
            rejected_rows.append(
                {
                    "Amount Requested": 6000 + index * 500 + (year - 2012) * 100,
                    "Application Date": f"{year}-0{(index % 9) + 1}",
                    "Loan Title": "debt_consolidation" if index % 2 else "medical",
                    "Risk_Score": 660 - index * 4,
                    "Debt-To-Income Ratio": 12 + index * 3,
                    "Zip Code": f"80{index:03d}",
                    "State": "TX" if index % 2 else "CA",
                    "Employment Length": index,
                    "Policy Code": 0,
                }
            )

    accepted_path = tmp_path / "accepted.csv"
    rejected_path = tmp_path / "rejected.csv"
    pd.DataFrame(accepted_rows).to_csv(accepted_path, index=False)
    pd.DataFrame(rejected_rows).to_csv(rejected_path, index=False)
    return accepted_path, rejected_path


def test_run_protocol_2_returns_future_accepted_metrics_and_rejected_diagnostics():
    x_train, y_train, x_test, y_test, x_rejected = make_protocol2_features()

    result = run_protocol_2(
        x_train,
        y_train,
        x_test,
        y_test,
        x_rejected,
        teacher_config={"n_models": 1, "model_types": ["mlp"]},
        student_model_type="logistic",
        tau_u=0.6,
        random_state=11,
    )

    assert result["model"].tolist() == ["UCRI-CS", "teacher", "accepted-only"]
    assert result["protocol"].eq("Protocol2").all()
    assert result["evaluation_population"].eq("future_accepted_test").all()
    assert not result["real_rejected_label_available"].any()
    assert {"AUROC", "Brier", "ECE", "real_rejected_mean_pd", "real_rejected_overlap_coverage"}.issubset(result.columns)
    assert "rejected_AUROC" not in result.columns
    assert result.loc[result["model"].eq("UCRI-CS"), "pseudo_label_coverage"].iloc[0] >= 0.0
    assert result["real_rejected_mmd_rbf"].ge(0).all()


def test_run_protocol_2_samples_large_rejected_diagnostics():
    x_train, y_train, x_test, y_test, x_rejected = make_protocol2_features(
        n_train=120,
        n_test=60,
        n_rejected=180,
    )

    result = run_protocol_2(
        x_train,
        y_train,
        x_test,
        y_test,
        x_rejected,
        teacher_config={"n_models": 1, "model_types": ["mlp"]},
        student_model_type="logistic",
        diagnostic_sample_size=25,
        random_state=11,
    )

    assert result["real_rejected_diagnostic_train_sample_size"].iloc[0] == 25
    assert result["real_rejected_diagnostic_rejected_sample_size"].iloc[0] == 25
    assert result["n_real_rejected_unlabeled"].iloc[0] == 180


def test_protocol2_main_writes_metrics_csv(tmp_path):
    accepted_path, rejected_path = make_raw_lendingclub_files(tmp_path)
    output_path = tmp_path / "protocol2_metrics.csv"

    result = main(
        str(accepted_path),
        str(rejected_path),
        str(output_path),
        teacher_config={"n_models": 1, "model_types": ["mlp"]},
        student_model_type="logistic",
        max_accepted_rows=None,
        max_rejected_rows=None,
        random_state=9,
    )

    saved = pd.read_csv(output_path)
    assert output_path.exists()
    assert saved["model"].tolist() == ["UCRI-CS", "teacher", "accepted-only"]
    assert result["n_real_rejected_unlabeled"].iloc[0] == 21
    assert saved["n_train_accepted"].iloc[0] == 24
    assert saved["n_test_accepted"].iloc[0] == 16
    assert {"AUROC", "PR-AUC", "KS", "Brier", "ECE"}.issubset(saved.columns)
    assert "real_rejected_hidden_bad_rate" not in saved.columns


def test_protocol2_main_applies_row_limits_before_feature_alignment(tmp_path):
    accepted_path, rejected_path = make_raw_lendingclub_files(tmp_path)
    output_path = tmp_path / "protocol2_limited_metrics.csv"

    result = main(
        str(accepted_path),
        str(rejected_path),
        str(output_path),
        teacher_config={"n_models": 1, "model_types": ["mlp"]},
        student_model_type="logistic",
        max_accepted_rows=12,
        max_rejected_rows=5,
        random_state=9,
    )

    assert result["n_train_accepted"].iloc[0] == 12
    assert result["n_test_accepted"].iloc[0] == 12
    assert result["n_real_rejected_unlabeled"].iloc[0] == 5


def test_protocol2_validates_inputs():
    x_train, y_train, x_test, y_test, x_rejected = make_protocol2_features(n_train=20, n_test=10, n_rejected=8)

    with pytest.raises(ValueError, match="same length"):
        run_protocol_2(x_train, y_train[:-1], x_test, y_test, x_rejected)

    with pytest.raises(ValueError, match="both classes"):
        run_protocol_2(x_train, np.zeros(len(y_train), dtype=int), x_test, y_test, x_rejected)

    with pytest.raises(ValueError, match="X_real_rejected"):
        run_protocol_2(x_train, y_train, x_test, y_test, x_rejected.iloc[0:0])
