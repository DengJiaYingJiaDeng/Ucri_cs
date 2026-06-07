from __future__ import annotations

import numpy as np
import pandas as pd


def hard_augmentation(
    model,
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
    threshold: float = 0.5,
):
    """Classic hard reject-inference augmentation.

    The accepted-only model scores rejected applicants, assigns hard pseudo-labels,
    then refits on accepted labels plus pseudo-labeled rejected rows.
    """
    x_labeled, y_labeled, x_unlabeled = _validate_training_inputs(X_labeled, y_labeled, X_unlabeled)
    threshold = _validate_probability_threshold("threshold", threshold)

    model.fit(x_labeled, y_labeled)
    probabilities = _predict_positive_probability(model, x_unlabeled)
    hard_labels = (probabilities >= threshold).astype(int)

    x_all = pd.concat([x_labeled, x_unlabeled], ignore_index=True)
    y_all = np.concatenate([y_labeled, hard_labels])
    model.fit(x_all, y_all)
    return model


def fuzzy_augmentation(
    model,
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
):
    """Fuzzy augmentation using model-estimated rejected bad-rate weights."""
    x_labeled, y_labeled, x_unlabeled = _validate_training_inputs(X_labeled, y_labeled, X_unlabeled)

    model.fit(x_labeled, y_labeled)
    probabilities = _predict_positive_probability(model, x_unlabeled)

    x_all, y_all, sample_weight = _expand_unlabeled_soft_labels(
        x_labeled,
        y_labeled,
        x_unlabeled,
        probabilities,
    )
    _fit_with_optional_sample_weight(model, x_all, y_all, sample_weight)
    return model


def parceling(
    model,
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
    n_bins: int = 10,
):
    """Risk-score parceling baseline with bin-level rejected bad-rate labels."""
    x_labeled, y_labeled, x_unlabeled = _validate_training_inputs(X_labeled, y_labeled, X_unlabeled)
    n_bins = _validate_n_bins(n_bins)

    model.fit(x_labeled, y_labeled)
    probabilities = _predict_positive_probability(model, x_unlabeled)
    parcel_probabilities = _compute_parcel_probabilities(probabilities, n_bins)

    x_all, y_all, sample_weight = _expand_unlabeled_soft_labels(
        x_labeled,
        y_labeled,
        x_unlabeled,
        parcel_probabilities,
    )
    _fit_with_optional_sample_weight(model, x_all, y_all, sample_weight)
    return model


def self_training(
    model,
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
    n_iterations: int = 3,
    confidence_threshold: float = 0.8,
):
    """Vanilla self-training baseline without uncertainty-aware filtering."""
    x_labeled, y_labeled, current_unlabeled = _validate_training_inputs(X_labeled, y_labeled, X_unlabeled)
    n_iterations = _validate_n_iterations(n_iterations)
    confidence_threshold = _validate_confidence_threshold(confidence_threshold)

    model.fit(x_labeled, y_labeled)
    for _ in range(n_iterations):
        if len(current_unlabeled) == 0:
            break

        probabilities = _predict_positive_probability(model, current_unlabeled)
        confident = (probabilities >= confidence_threshold) | (probabilities <= 1.0 - confidence_threshold)
        if not confident.any():
            break

        hard_labels = (probabilities[confident] >= 0.5).astype(int)
        x_labeled = pd.concat([x_labeled, current_unlabeled.iloc[np.flatnonzero(confident)]], ignore_index=True)
        y_labeled = np.concatenate([y_labeled, hard_labels])
        current_unlabeled = current_unlabeled.iloc[np.flatnonzero(~confident)].reset_index(drop=True)
        model.fit(x_labeled, y_labeled)

    return model


def ipw_weighted_pd(
    propensity_model,
    pd_model,
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    eps: float = 0.01,
):
    """Accepted-only PD baseline reweighted by inverse approval propensity."""
    x_labeled = pd.DataFrame(X_labeled).copy().reset_index(drop=True)
    y_labeled = _validate_binary_labels(y_labeled, expected_length=len(x_labeled))
    eps = _validate_positive_eps(eps)

    propensity = _predict_propensity(propensity_model, x_labeled)
    weights = 1.0 / np.maximum(propensity, eps)
    _fit_with_optional_sample_weight(pd_model, x_labeled, y_labeled, weights)
    return pd_model


def extrapolation_reject_inference(
    model,
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
    quantile: float = 0.05,
):
    """Extrapolation baseline that adds highest-risk rejected rows as bad labels."""
    x_labeled, y_labeled, x_unlabeled = _validate_training_inputs(X_labeled, y_labeled, X_unlabeled)
    quantile = _validate_quantile(quantile)

    model.fit(x_labeled, y_labeled)
    probabilities = _predict_positive_probability(model, x_unlabeled)
    high_risk_threshold = float(np.quantile(probabilities, 1.0 - quantile))
    high_risk_mask = probabilities >= high_risk_threshold
    if not high_risk_mask.any():
        return model

    high_risk_rejected = x_unlabeled.iloc[np.flatnonzero(high_risk_mask)].reset_index(drop=True)
    x_all = pd.concat([x_labeled, high_risk_rejected], ignore_index=True)
    y_all = np.concatenate([y_labeled, np.ones(len(high_risk_rejected), dtype=int)])
    model.fit(x_all, y_all)
    return model


def domain_adversarial_balancing(
    pd_model,
    X_accepted: pd.DataFrame,
    y_accepted: np.ndarray,
    X_rejected: pd.DataFrame,
    n_epochs: int = 50,
):
    """Simplified domain-adversarial baseline via accepted/rejected reweighting."""
    x_accepted, y_accepted, x_rejected = _validate_training_inputs(X_accepted, y_accepted, X_rejected)
    _validate_non_negative_int("n_epochs", n_epochs)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    x_all = pd.concat([x_accepted, x_rejected], ignore_index=True)
    domain_labels = np.concatenate(
        [
            np.zeros(len(x_accepted), dtype=int),
            np.ones(len(x_rejected), dtype=int),
        ]
    )
    discriminator = Pipeline(
        steps=[
            ("preprocessor", _build_feature_preprocessor(x_all)),
            ("estimator", LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")),
        ]
    )
    discriminator.fit(x_all, domain_labels)

    rejected_domain_probability = _predict_positive_probability(discriminator, x_accepted)
    weights = rejected_domain_probability / np.clip(1.0 - rejected_domain_probability, 1e-6, None)
    weights = np.clip(weights, 0.1, 10.0)
    _fit_with_optional_sample_weight(pd_model, x_accepted, y_accepted, weights)
    return pd_model


def ssvm_reject_inference(
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
    n_neighbors: int = 7,
    random_state: int = 42,
):
    """Semi-supervised SVM baseline using label propagation pseudo-labels."""
    x_labeled, y_labeled, x_unlabeled = _validate_training_inputs(X_labeled, y_labeled, X_unlabeled)
    _require_both_classes("y_labeled", y_labeled)
    x_all = pd.concat([x_labeled, x_unlabeled], ignore_index=True)
    n_neighbors = _validate_n_neighbors(n_neighbors, max_neighbors=len(x_all) - 1)

    from sklearn.pipeline import Pipeline
    from sklearn.semi_supervised import LabelPropagation
    from sklearn.svm import SVC

    preprocessor = _build_feature_preprocessor(x_all)
    x_processed = preprocessor.fit_transform(x_all)
    semi_supervised_labels = np.concatenate([y_labeled, np.full(len(x_unlabeled), -1, dtype=int)])
    label_propagation = LabelPropagation(kernel="knn", n_neighbors=n_neighbors)
    propagated_labels = label_propagation.fit(x_processed, semi_supervised_labels).transduction_.astype(int)
    propagated_labels[propagated_labels == -1] = 0

    svm = Pipeline(
        steps=[
            ("preprocessor", _build_feature_preprocessor(x_all)),
            ("estimator", SVC(probability=True, kernel="rbf", random_state=random_state)),
        ]
    )
    svm.fit(x_all, propagated_labels)
    return svm


def mean_teacher_baseline(
    student_model,
    teacher_model,
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
    n_iterations: int = 5,
    ema_decay: float = 0.99,
):
    """Supplementary Mean Teacher-style tabular baseline.

    Generic sklearn estimators do not expose weights for a true EMA update, so
    this baseline uses teacher soft targets on rejected rows as the consistency
    signal and falls back to weighted soft-label expansion when needed.
    """
    x_labeled, y_labeled, x_unlabeled = _validate_training_inputs(X_labeled, y_labeled, X_unlabeled)
    n_iterations = _validate_n_iterations(n_iterations)
    _validate_ema_decay(ema_decay)

    for _ in range(n_iterations):
        teacher_probs = _predict_positive_probability(teacher_model, x_unlabeled)
        try:
            student_model.fit(x_labeled, y_labeled, x_unlabeled, teacher_probs, lambda_distill=0.5)
        except TypeError:
            x_all, y_all, sample_weight = _expand_unlabeled_soft_labels(
                x_labeled,
                y_labeled,
                x_unlabeled,
                teacher_probs,
            )
            sample_weight[len(x_labeled) :] *= 0.5
            _fit_with_optional_sample_weight(student_model, x_all, y_all, sample_weight)
    return student_model


def noisy_student_baseline(
    student_model,
    teacher_model,
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
    noise_std: float = 0.05,
    n_iterations: int = 3,
    random_state: int = 42,
):
    """Supplementary Noisy Student-style tabular baseline."""
    x_labeled, y_labeled, x_unlabeled = _validate_training_inputs(X_labeled, y_labeled, X_unlabeled)
    noise_std = _validate_noise_std(noise_std)
    n_iterations = _validate_n_iterations(n_iterations)
    rng = np.random.default_rng(random_state)

    for _ in range(n_iterations):
        teacher_probs = _predict_positive_probability(teacher_model, x_unlabeled)
        pseudo_labels = (teacher_probs >= 0.5).astype(int)
        x_noisy = x_unlabeled.copy()
        numeric_columns = x_noisy.select_dtypes(include=[np.number]).columns
        if len(numeric_columns):
            noise = rng.normal(0.0, noise_std, size=x_noisy[numeric_columns].shape)
            x_noisy.loc[:, numeric_columns] = x_noisy[numeric_columns].to_numpy(dtype=float) + noise

        x_all = pd.concat([x_labeled, x_noisy], ignore_index=True)
        y_all = np.concatenate([y_labeled, pseudo_labels])
        student_model.fit(x_all, y_all)
    return student_model


def _validate_training_inputs(
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    x_labeled = pd.DataFrame(X_labeled).copy().reset_index(drop=True)
    x_unlabeled = pd.DataFrame(X_unlabeled).copy().reset_index(drop=True)
    y_labeled = _validate_binary_labels(y_labeled, expected_length=len(x_labeled))

    if len(x_labeled) == 0:
        raise ValueError("X_labeled must not be empty.")
    if len(x_unlabeled) == 0:
        raise ValueError("X_unlabeled must not be empty.")

    x_unlabeled = x_unlabeled.reindex(columns=x_labeled.columns)
    return x_labeled, y_labeled, x_unlabeled


def _validate_binary_labels(labels: np.ndarray, expected_length: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("y_labeled must be a one-dimensional array.")
    if len(labels) != expected_length:
        raise ValueError("X_labeled and y_labeled must have the same length.")
    if len(labels) == 0:
        raise ValueError("y_labeled must not be empty.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_labeled must contain binary 0/1 labels.")
    return labels.astype(int)


def _validate_probability_threshold(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"{name} must be in [0, 1].")
    return value


def _validate_confidence_threshold(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.5 or value > 1.0:
        raise ValueError("confidence_threshold must be in [0.5, 1.0].")
    return value


def _validate_n_bins(n_bins: int) -> int:
    if isinstance(n_bins, bool):
        raise ValueError("n_bins must be a positive integer.")
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    return n_bins


def _validate_n_iterations(n_iterations: int) -> int:
    return _validate_non_negative_int("n_iterations", n_iterations)


def _validate_positive_eps(eps: float) -> float:
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive.")
    return eps


def _predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.ndim == 2:
        if probabilities.shape[1] < 2:
            raise ValueError("predict_proba must return a positive-class probability column.")
        probabilities = probabilities[:, 1]
    probabilities = _validate_probability_vector(probabilities, expected_length=len(X), name="predicted probabilities")
    return np.clip(probabilities, 0.0, 1.0)


def _predict_propensity(propensity_model, X: pd.DataFrame) -> np.ndarray:
    propensity = np.asarray(propensity_model.predict_proba(X), dtype=float)
    if propensity.ndim == 2:
        if propensity.shape[1] < 2:
            raise ValueError("propensity predict_proba must return a positive-class probability column.")
        propensity = propensity[:, 1]
    propensity = _validate_probability_vector(propensity, expected_length=len(X), name="propensity")
    return np.clip(propensity, 0.0, 1.0)


def _validate_probability_vector(values: np.ndarray, expected_length: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if len(values) != expected_length:
        raise ValueError(f"{name} must have the same length as X.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values.")
    return values


def _compute_parcel_probabilities(probabilities: np.ndarray, n_bins: int) -> np.ndarray:
    n_bins = min(n_bins, len(probabilities))
    ranks = pd.Series(probabilities).rank(method="first").to_numpy() - 1
    bin_ids = np.floor(ranks * n_bins / len(probabilities)).astype(int)
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    parcel_probabilities = np.zeros(len(probabilities), dtype=float)
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.any():
            parcel_probabilities[mask] = probabilities[mask].mean()
    return np.clip(parcel_probabilities, 0.0, 1.0)


def _expand_unlabeled_soft_labels(
    X_labeled: pd.DataFrame,
    y_labeled: np.ndarray,
    X_unlabeled: pd.DataFrame,
    probabilities: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    probabilities = _validate_probability_vector(probabilities, expected_length=len(X_unlabeled), name="soft labels")
    probabilities = np.clip(probabilities, 0.0, 1.0)

    x_all = pd.concat([X_labeled, X_unlabeled, X_unlabeled], ignore_index=True)
    y_all = np.concatenate(
        [
            y_labeled,
            np.zeros(len(X_unlabeled), dtype=int),
            np.ones(len(X_unlabeled), dtype=int),
        ]
    )
    sample_weight = np.concatenate(
        [
            np.ones(len(X_labeled), dtype=float),
            1.0 - probabilities,
            probabilities,
        ]
    )
    return x_all, y_all, sample_weight


def _fit_with_optional_sample_weight(model, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray) -> None:
    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)


def _validate_quantile(quantile: float) -> float:
    quantile = float(quantile)
    if not np.isfinite(quantile) or quantile <= 0.0 or quantile > 1.0:
        raise ValueError("quantile must be in (0, 1].")
    return quantile


def _validate_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _validate_n_neighbors(n_neighbors: int, max_neighbors: int) -> int:
    if isinstance(n_neighbors, bool):
        raise ValueError("n_neighbors must be a positive integer.")
    n_neighbors = int(n_neighbors)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer.")
    return min(n_neighbors, max_neighbors)


def _validate_noise_std(noise_std: float) -> float:
    noise_std = float(noise_std)
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")
    return noise_std


def _validate_ema_decay(ema_decay: float) -> float:
    ema_decay = float(ema_decay)
    if not np.isfinite(ema_decay) or ema_decay < 0.0 or ema_decay > 1.0:
        raise ValueError("ema_decay must be in [0, 1].")
    return ema_decay


def _require_both_classes(name: str, values: np.ndarray) -> None:
    if len(np.unique(values)) != 2:
        raise ValueError(f"{name} must contain both classes.")


def _build_feature_preprocessor(X: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    X = pd.DataFrame(X)
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("Feature frame must contain at least one column.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


REJECT_INFERENCE_BASELINES = {
    "hard": hard_augmentation,
    "fuzzy": fuzzy_augmentation,
    "parceling": parceling,
    "self_training": self_training,
    "ipw": ipw_weighted_pd,
    "extrapolation": extrapolation_reject_inference,
    "domain_adversarial": domain_adversarial_balancing,
    "ssvm": ssvm_reject_inference,
    "mean_teacher": mean_teacher_baseline,
    "noisy_student": noisy_student_baseline,
}
