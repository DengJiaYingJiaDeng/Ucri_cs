import numpy as np
import pandas as pd
import pytest

from experiments.confounded_simulation import (
    confounded_rejection,
    generate_hidden_confounder,
    main,
    run_confounded_experiment,
)


def make_confounded_data(n=180):
    rank = np.arange(n)
    return (
        pd.DataFrame(
            {
                "loan_amount": 3000 + rank * 75,
                "dti": 5 + (rank % 60) * 0.5,
                "fico_range_low": 760 - (rank % 90),
                "state": np.where(rank % 3 == 0, "CA", "NY"),
                "application_date": pd.date_range("2013-01-01", periods=n, freq="MS").astype(str),
            }
        ),
        (rank % 2).astype(int),
    )


def test_generate_hidden_confounder_separation_increases_with_rho():
    _, y = make_confounded_data()

    z_low = generate_hidden_confounder(y, rho=0.0, random_state=7)
    z_high = generate_hidden_confounder(y, rho=0.8, random_state=7)

    low_gap = z_low[y == 1].mean() - z_low[y == 0].mean()
    high_gap = z_high[y == 1].mean() - z_high[y == 0].mean()
    assert high_gap > low_gap + 0.5
    assert np.isclose(z_high.mean(), 0.0)


def test_confounded_rejection_targets_rate_and_preserves_hidden_labels():
    X, y = make_confounded_data()
    g_logits = np.linspace(-1.0, 1.0, len(y))

    result = confounded_rejection(
        X,
        y,
        g_logits,
        rho=0.5,
        confounder_gamma=1.0,
        rejection_rate=0.4,
        random_state=9,
    )

    accepted_mask = result["accepted_mask"]
    assert accepted_mask.dtype == bool
    assert accepted_mask.shape == (len(y),)
    assert len(result["X_acc"]) + len(result["X_rej"]) == len(X)
    assert np.array_equal(result["y_rej_hidden"], y[~accepted_mask])
    assert result["actual_rejection_rate"] == pytest.approx(0.4, abs=1 / len(y))
    assert np.all((result["propensity"] >= 0.01) & (result["propensity"] <= 0.99))


def test_hidden_confounder_changes_selection_when_gamma_is_positive():
    X, y = make_confounded_data()
    g_logits = np.zeros(len(y))

    unconfounded = confounded_rejection(
        X,
        y,
        g_logits,
        rho=0.8,
        confounder_gamma=0.0,
        rejection_rate=0.4,
        random_state=13,
    )
    confounded = confounded_rejection(
        X,
        y,
        g_logits,
        rho=0.8,
        confounder_gamma=2.0,
        rejection_rate=0.4,
        random_state=13,
    )

    unconfounded_gap = (
        unconfounded["z"][~unconfounded["accepted_mask"]].mean()
        - unconfounded["z"][unconfounded["accepted_mask"]].mean()
    )
    confounded_gap = (
        confounded["z"][~confounded["accepted_mask"]].mean()
        - confounded["z"][confounded["accepted_mask"]].mean()
    )
    assert confounded_gap > unconfounded_gap + 0.5


def test_confounded_simulation_validates_inputs():
    X, y = make_confounded_data()
    g_logits = np.zeros(len(y))

    with pytest.raises(ValueError, match="rho"):
        generate_hidden_confounder(y, rho=1.1)

    with pytest.raises(ValueError, match="same length"):
        confounded_rejection(X, y, g_logits[:-1], rho=0.2, confounder_gamma=1.0, rejection_rate=0.4)

    with pytest.raises(ValueError, match="confounder_gamma"):
        confounded_rejection(X, y, g_logits, rho=0.2, confounder_gamma=-1.0, rejection_rate=0.4)

    with pytest.raises(ValueError, match="rejection_rate"):
        confounded_rejection(X, y, g_logits, rho=0.2, confounder_gamma=1.0, rejection_rate=0.0)


def test_run_confounded_experiment_returns_model_metric_rows():
    X, y = make_confounded_data()

    result = run_confounded_experiment(
        X,
        y,
        rho_values=[0.0],
        confounder_gammas=[0.5],
        rejection_rates=[0.3],
        teacher_config={"n_models": 1, "model_types": ["lightgbm"]},
        student_model_type="lightgbm",
        random_state=7,
    )

    assert result["model"].tolist() == ["UCRI-CS", "teacher", "accepted-only"]
    assert result["protocol"].eq("ConfoundedSimulation").all()
    assert {"AUROC", "PR-AUC", "KS", "Brier", "ECE", "pseudo_label_coverage"}.issubset(result.columns)
    assert result["n_rejected"].iloc[0] == 54


def test_confounded_main_writes_metrics_csv(tmp_path):
    X, y = make_confounded_data(n=90)
    raw = X.rename(columns={"loan_amount": "loan_amnt", "application_date": "issue_d"}).copy()
    raw["loan_status"] = np.where(y == 1, "Charged Off", "Fully Paid")

    data_path = tmp_path / "accepted.csv"
    output_path = tmp_path / "confounded_metrics.csv"
    raw.to_csv(data_path, index=False)

    result = main(
        str(data_path),
        str(output_path),
        rho_values=[0.0],
        confounder_gammas=[0.5],
        rejection_rates=[0.3],
        teacher_config={"n_models": 1, "model_types": ["lightgbm"]},
        student_model_type="lightgbm",
        max_rows=90,
        random_state=7,
    )

    saved = pd.read_csv(output_path)
    assert output_path.exists()
    assert result["model"].tolist() == ["UCRI-CS", "teacher", "accepted-only"]
    assert saved["model"].tolist() == ["UCRI-CS", "teacher", "accepted-only"]
