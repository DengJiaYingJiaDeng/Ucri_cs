import numpy as np
import pandas as pd
import pytest

from experiments.protocol3_simulated_rejection import (
    SIMULATION_MECHANISMS,
    compute_rejection_distribution_comparison,
    main,
    run_protocol_3,
    simulate_rejection,
)


def make_simulated_data(n=160):
    rank = np.arange(n)
    return (
        pd.DataFrame(
            {
                "loan_amnt": 3000 + rank * 80,
                "dti": 5 + (rank % 70) * 0.5,
                "fico_avg": 760 - (rank % 80),
                "int_rate": 6.0 + (rank % 30) * 0.25,
                "state": np.where(rank % 4 < 2, "CA", "NY"),
                "application_date": pd.date_range("2013-01-01", periods=n, freq="MS").astype(str),
            }
        ),
        (rank % 2).astype(int),
    )


def test_simulate_rejection_preserves_hidden_rejected_labels_for_all_mechanisms():
    X, y = make_simulated_data()

    for mechanism in SIMULATION_MECHANISMS:
        accepted_mask, X_accepted, y_accepted, X_rejected, y_rejected_hidden = simulate_rejection(
            X,
            y,
            mechanism=mechanism,
            rejection_rate=0.35,
            overlap_level="medium",
            policy_noise=0.1,
            random_state=11,
        )

        assert accepted_mask.dtype == bool
        assert accepted_mask.shape == (len(X),)
        assert len(X_accepted) + len(X_rejected) == len(X)
        assert len(y_accepted) + len(y_rejected_hidden) == len(y)
        assert np.array_equal(y_rejected_hidden, y[~accepted_mask])
        assert X_accepted.columns.tolist() == X.columns.tolist()
        assert X_rejected.columns.tolist() == X.columns.tolist()


def test_simulate_rejection_validates_options():
    X, y = make_simulated_data()

    with pytest.raises(ValueError, match="mechanism"):
        simulate_rejection(X, y, mechanism="missing")

    with pytest.raises(ValueError, match="rejection_rate"):
        simulate_rejection(X, y, rejection_rate=1.0)

    with pytest.raises(ValueError, match="overlap_level"):
        simulate_rejection(X, y, overlap_level="none")

    with pytest.raises(ValueError, match="policy_noise"):
        simulate_rejection(X, y, policy_noise=-0.1)


def test_distribution_comparison_returns_mmd_and_distance():
    X, y = make_simulated_data()
    _, X_accepted, _, X_rejected, _ = simulate_rejection(
        X,
        y,
        mechanism="rule_based",
        rejection_rate=0.3,
        random_state=5,
    )

    comparison = compute_rejection_distribution_comparison(X_accepted, X_rejected, max_samples=80)

    assert comparison["mmd_rbf"] >= 0.0
    assert comparison["mean_pairwise_distance"] >= 0.0
    assert np.isfinite(comparison["mmd_rbf"])
    assert np.isfinite(comparison["mean_pairwise_distance"])


def test_run_protocol_3_returns_model_metric_rows():
    X, y = make_simulated_data()

    result = run_protocol_3(
        X,
        y,
        mechanisms=["rule_based"],
        rejection_rates=[0.3],
        overlap_levels=["medium"],
        policy_noises=[0.2],
        teacher_config={"n_models": 1, "model_types": ["mlp"]},
        student_model_type="logistic",
        random_state=7,
    )

    assert result["model"].tolist() == ["UCRI-CS", "teacher", "accepted-only"]
    assert {"AUROC", "PR-AUC", "KS", "Brier", "ECE", "mmd_rbf"}.issubset(result.columns)
    assert result["protocol"].eq("Protocol3").all()
    assert result["n_rejected"].iloc[0] == 48


def test_protocol3_main_writes_metrics_csv(tmp_path):
    X, y = make_simulated_data(n=90)
    raw = X.rename(columns={"application_date": "issue_d"}).copy()
    raw["loan_status"] = np.where(y == 1, "Charged Off", "Fully Paid")

    data_path = tmp_path / "accepted.csv"
    output_path = tmp_path / "protocol3_metrics.csv"
    raw.to_csv(data_path, index=False)

    result = main(
        str(data_path),
        str(output_path),
        mechanisms=["rule_based"],
        rejection_rates=[0.3],
        overlap_levels=["medium"],
        policy_noises=[0.2],
        teacher_config={"n_models": 1, "model_types": ["mlp"]},
        student_model_type="logistic",
        max_rows=90,
        random_state=7,
    )

    saved = pd.read_csv(output_path)
    assert output_path.exists()
    assert result["model"].tolist() == ["UCRI-CS", "teacher", "accepted-only"]
    assert saved["model"].tolist() == ["UCRI-CS", "teacher", "accepted-only"]
