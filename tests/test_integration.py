import numpy as np
import pandas as pd
import pytest

from src.data.leakage_audit import ForbiddenFeatureError, audit_features
from src.data.preprocess import construct_default_label, label_maturity_filter
from src.evaluation.metrics import compute_all_metrics
from src.models.propensity import PropensityModel
from src.reject_inference.ssl_trainer import UCRITrainer


def _make_synthetic_data(n_accepted: int = 260, n_rejected: int = 90, seed: int = 42):
    rng = np.random.default_rng(seed)
    x_accepted = pd.DataFrame(
        {
            "loan_amount": rng.lognormal(9.2, 0.45, n_accepted),
            "dti": rng.uniform(5, 42, n_accepted),
            "emp_length": rng.integers(0, 30, n_accepted),
            "fico_avg": rng.normal(685, 45, n_accepted),
            "state": rng.choice(["CA", "NY", "TX", "FL"], n_accepted),
            "loan_purpose": rng.choice(["debt_consolidation", "credit_card", "home_improvement"], n_accepted),
        }
    )
    accepted_risk = (
        0.000035 * x_accepted["loan_amount"]
        + 0.065 * x_accepted["dti"]
        - 0.010 * x_accepted["fico_avg"]
        - 0.020 * x_accepted["emp_length"]
        + x_accepted["state"].map({"CA": 0.15, "NY": -0.05, "TX": 0.10, "FL": 0.00}).to_numpy()
    )
    y_accepted = (accepted_risk > np.median(accepted_risk)).astype(int).to_numpy()

    x_rejected = pd.DataFrame(
        {
            "loan_amount": rng.lognormal(9.45, 0.55, n_rejected),
            "dti": rng.uniform(12, 50, n_rejected),
            "emp_length": rng.integers(0, 25, n_rejected),
            "fico_avg": rng.normal(655, 55, n_rejected),
            "state": rng.choice(["CA", "NY", "TX", "FL"], n_rejected),
            "loan_purpose": rng.choice(["debt_consolidation", "small_business", "medical"], n_rejected),
        }
    )
    return x_accepted, y_accepted, x_rejected


def _make_lendingclub_status_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off", "Current", "Default"],
            "loan_amnt": [10000, 15000, 8000, 20000],
            "dti": [12.0, 30.0, 18.0, 38.0],
        }
    )


def _lightgbm_teacher_config(n_models: int = 2):
    return {"n_models": n_models, "model_types": ["lightgbm"] * n_models}


def test_ucri_cs_pipeline_integration():
    x_accepted, y_accepted, x_rejected = _make_synthetic_data()
    audit_features(x_accepted)
    trainer = UCRITrainer(
        teacher_config=_lightgbm_teacher_config(),
        student_model_type="lightgbm",
        tau_u=0.6,
        gamma=2.0,
        lambda_distill=0.3,
        random_state=42,
    )

    result = trainer.run(x_accepted, y_accepted, x_rejected)
    student_predictions = result["student"].predict_proba(x_accepted)
    metrics = compute_all_metrics(y_accepted, student_predictions)

    assert {"teacher", "student", "pseudo_labels", "uncertainty", "teacher_probs"}.issubset(result)
    assert student_predictions.shape == (len(x_accepted),)
    assert result["teacher_probs"].shape == (len(x_rejected),)
    assert result["pseudo_labels"]["soft_label"].shape == (len(x_rejected),)
    assert result["uncertainty"].shape == (len(x_rejected),)
    assert np.all((student_predictions >= 0.0) & (student_predictions <= 1.0))
    assert metrics["AUROC"] > 0.85
    assert metrics["Brier"] < 0.25


def test_pipeline_with_propensity_weights_and_label_audit():
    status_frame = construct_default_label(label_maturity_filter(_make_lendingclub_status_frame()))
    assert status_frame["default_label"].tolist() == [0.0, 1.0, 1.0]
    with pytest.raises(ForbiddenFeatureError):
        audit_features(status_frame[["loan_status", "loan_amnt"]])

    x_accepted, y_accepted, x_rejected = _make_synthetic_data(n_accepted=220, n_rejected=80, seed=7)
    propensity_model = PropensityModel(model_type="lightgbm", random_state=7)
    x_all = pd.concat([x_accepted, x_rejected], ignore_index=True)
    accepted_indicator = np.concatenate([np.ones(len(x_accepted), dtype=int), np.zeros(len(x_rejected), dtype=int)])
    propensity_model.fit(x_all, accepted_indicator)
    propensity_weights = propensity_model.compute_weights(x_accepted)

    trainer = UCRITrainer(
        teacher_config=_lightgbm_teacher_config(),
        student_model_type="lightgbm",
        tau_u=0.6,
        gamma=1.0,
        lambda_distill=0.2,
        random_state=7,
    )
    result = trainer.run(x_accepted, y_accepted, x_rejected)
    student_predictions = result["student"].predict_proba(x_accepted)
    metrics = compute_all_metrics(y_accepted, student_predictions)

    assert propensity_weights.shape == (len(x_accepted),)
    assert np.all(np.isfinite(propensity_weights))
    assert np.all(propensity_weights >= 1.0)
    assert result["student"].training_targets_ is not None
    assert len(result["student"].training_targets_) == len(x_accepted) + len(x_rejected)
    assert metrics["AUROC"] > 0.80
