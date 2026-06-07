import numpy as np
import pandas as pd
import pytest

from src.models.student import StudentModel
from src.models.teacher import TeacherEnsemble
from src.reject_inference.ssl_trainer import UCRITrainer


@pytest.fixture
def trainer_data():
    rng = np.random.default_rng(42)
    n_labeled = 180
    n_rejected = 70
    x_labeled = pd.DataFrame(
        {
            "loan_amount": rng.lognormal(9, 0.5, n_labeled),
            "dti": rng.uniform(5, 40, n_labeled),
            "emp_length": rng.integers(0, 30, n_labeled),
            "fico_avg": rng.normal(680, 40, n_labeled),
        }
    )
    logit = -3.0 + 0.25 * np.log(x_labeled["loan_amount"]) + 0.035 * x_labeled["dti"] - 0.004 * x_labeled["fico_avg"]
    prob = 1 / (1 + np.exp(-logit))
    y_labeled = rng.binomial(1, prob).astype(int)

    x_rejected = pd.DataFrame(
        {
            "loan_amount": rng.lognormal(9.2, 0.6, n_rejected),
            "dti": rng.uniform(10, 45, n_rejected),
            "emp_length": rng.integers(0, 25, n_rejected),
            "fico_avg": rng.normal(660, 50, n_rejected),
        }
    )
    return x_labeled, y_labeled, x_rejected


def _lightgbm_teacher_config(n_models: int = 2):
    return {"n_models": n_models, "model_types": ["lightgbm"] * n_models}


def test_trainer_runs_full_pipeline(trainer_data):
    x, y, x_rej = trainer_data
    trainer = UCRITrainer(
        teacher_config=_lightgbm_teacher_config(),
        student_model_type="lightgbm",
        tau_u=0.6,
    )

    result = trainer.run(x, y, x_rej)

    assert {"teacher", "student", "pseudo_labels", "uncertainty", "teacher_probs"}.issubset(result)
    assert isinstance(result["teacher"], TeacherEnsemble)
    assert isinstance(result["student"], StudentModel)
    assert len(result["pseudo_labels"]["soft_label"]) == len(x_rej)
    assert len(result["uncertainty"]) == len(x_rej)


def test_trainer_outputs_valid_predictions(trainer_data):
    x, y, x_rej = trainer_data
    trainer = UCRITrainer(
        teacher_config=_lightgbm_teacher_config(),
        student_model_type="lightgbm",
        tau_u=0.6,
    )

    result = trainer.run(x, y, x_rej)
    preds = result["student"].predict_proba(x)

    assert preds.shape == (len(x),)
    assert np.all((preds >= 0) & (preds <= 1))
    assert np.all((result["teacher_probs"] >= 0) & (result["teacher_probs"] <= 1))
    assert np.all((result["uncertainty"] >= 0) & (result["uncertainty"] <= 1))


def test_trainer_calibrates_teacher_when_calibration_data_provided(trainer_data):
    x, y, x_rej = trainer_data
    trainer = UCRITrainer(
        teacher_config=_lightgbm_teacher_config(),
        student_model_type="logistic",
        tau_u=0.6,
    )

    result = trainer.run(x.iloc[:120], y[:120], x_rej, X_calib=x.iloc[120:], y_calib=y[120:])

    assert result["teacher"].calibrated
    assert 0.01 <= result["teacher"].temperature <= 10.0


def test_trainer_out_of_fold_produces_oof_probs(trainer_data):
    x, y, x_rej = trainer_data
    trainer = UCRITrainer(
        teacher_config=_lightgbm_teacher_config(),
        student_model_type="lightgbm",
        tau_u=0.6,
    )

    result = trainer.run_out_of_fold(x, y, x_rej, n_folds=3)

    assert "oof_probs" in result
    assert len(result["oof_probs"]) == len(x)
    assert np.all((result["oof_probs"] >= 0) & (result["oof_probs"] <= 1))
    assert np.all(np.isfinite(result["oof_probs"]))
    assert len(result["teacher_probs"]) == len(x_rej)


def test_trainer_out_of_fold_preserves_student_training_targets(trainer_data):
    x, y, x_rej = trainer_data
    trainer = UCRITrainer(
        teacher_config=_lightgbm_teacher_config(),
        student_model_type="logistic",
        lambda_distill=0.5,
        tau_u=0.6,
    )

    result = trainer.run_out_of_fold(x, y, x_rej, n_folds=3)

    assert result["student"].training_targets_ is not None
    assert len(result["student"].training_targets_) == len(x) + len(x_rej)
    assert np.allclose(result["student"].training_targets_[len(x) :], result["pseudo_labels"]["soft_label"])


def test_trainer_rejects_mismatched_lengths(trainer_data):
    x, y, x_rej = trainer_data
    trainer = UCRITrainer(teacher_config=_lightgbm_teacher_config(), student_model_type="lightgbm")

    with pytest.raises(ValueError, match="same length"):
        trainer.run(x, y[:-1], x_rej)


def test_trainer_rejects_too_many_folds_for_class_counts(trainer_data):
    x, y, x_rej = trainer_data
    y = np.array([0] * (len(y) - 1) + [1])
    trainer = UCRITrainer(teacher_config=_lightgbm_teacher_config(), student_model_type="lightgbm")

    with pytest.raises(ValueError, match="n_folds"):
        trainer.run_out_of_fold(x, y, x_rej, n_folds=3)
