import numpy as np
import pandas as pd
import pytest

from src.evaluation.protocol import ProtocolResult, run_protocol_1


def make_protocol_data():
    rng = np.random.default_rng(42)
    n_train = 80
    n_val = 30
    n_test = 40

    def frame(n, shift):
        score = rng.normal(0.0 + shift, 1.0, n)
        dti = rng.uniform(5, 35, n)
        x = pd.DataFrame({"score": score, "dti": dti})
        probability = 1 / (1 + np.exp(-(score + 0.04 * dti - 0.8)))
        y = rng.binomial(1, probability)
        return x, y

    return (*frame(n_train, 0.0), *frame(n_val, 0.2), *frame(n_test, 0.4))


def test_run_protocol_1_returns_result_objects():
    X_train, y_train, X_val, y_val, X_test, y_test = make_protocol_data()

    results = run_protocol_1(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        model_names=["LogisticRegression"],
        random_state=7,
    )

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, ProtocolResult)
    assert result.protocol == "Protocol1"
    assert result.model_name == "LogisticRegression"
    assert result.predictions.shape == (len(X_test),)
    assert np.array_equal(result.true_labels, y_test)
    assert {"AUROC", "PR-AUC", "KS", "Brier", "ECE"}.issubset(result.metrics)


def test_run_protocol_1_supports_multiple_model_names():
    X_train, y_train, X_val, y_val, X_test, y_test = make_protocol_data()

    results = run_protocol_1(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        model_names=["LogisticRegression", "RandomForest"],
        random_state=7,
    )

    assert [result.model_name for result in results] == ["LogisticRegression", "RandomForest"]


def test_run_protocol_1_validates_inputs():
    X_train, y_train, X_val, y_val, X_test, y_test = make_protocol_data()

    with pytest.raises(ValueError, match="same length"):
        run_protocol_1(X_train, y_train[:-1], X_val, y_val, X_test, y_test)

    with pytest.raises(KeyError, match="Unknown model"):
        run_protocol_1(X_train, y_train, X_val, y_val, X_test, y_test, model_names=["MissingModel"])

    with pytest.raises(ValueError, match="binary"):
        run_protocol_1(X_train, np.full(len(y_train), 2), X_val, y_val, X_test, y_test)
