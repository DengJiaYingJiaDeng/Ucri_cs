from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data.leakage_audit import audit_features
from src.data.loader import load_accepted
from src.data.preprocess import build_shared_features, construct_default_label, label_maturity_filter
from src.evaluation.metrics import compute_all_metrics
from src.models.student import StudentModel
from src.reject_inference.ssl_trainer import UCRITrainer


DEFAULT_RHO_VALUES = [0.0, 0.2, 0.4, 0.6]
DEFAULT_CONFOUNDER_GAMMAS = [0.0, 0.5, 1.0, 2.0]
DEFAULT_REJECTION_RATES = [0.2, 0.4, 0.6]
DEFAULT_TEACHER_CONFIG = {"n_models": 3, "model_types": ["lightgbm", "catboost", "mlp"]}


def generate_hidden_confounder(y: np.ndarray, rho: float, random_state: int = 42) -> np.ndarray:
    """Generate an unobserved factor correlated with hidden repayment outcome."""
    labels = _validate_binary_labels(y)
    rho = _validate_rho(rho)
    rng = np.random.default_rng(random_state)

    centered_y = labels.astype(float) - labels.mean()
    y_std = centered_y.std()
    if y_std > 0:
        centered_y = centered_y / y_std
    epsilon = rng.normal(0.0, 1.0, len(labels))
    z = rho * centered_y + np.sqrt(max(0.0, 1.0 - rho**2)) * epsilon
    z_std = z.std()
    if z_std > 0:
        z = (z - z.mean()) / z_std
    return z.astype(float)


def confounded_rejection(
    X: pd.DataFrame,
    y: np.ndarray,
    g_logits: np.ndarray,
    rho: float,
    confounder_gamma: float,
    rejection_rate: float,
    random_state: int = 42,
) -> dict[str, object]:
    """Simulate rejection when approval depends on observed policy score and hidden confounder."""
    x, labels = _validate_feature_label_pair(X, y)
    g_logits = _validate_g_logits(g_logits, expected_length=len(x))
    rho = _validate_rho(rho)
    confounder_gamma = _validate_non_negative("confounder_gamma", confounder_gamma)
    rejection_rate = _validate_rejection_rate(rejection_rate)

    rng = np.random.default_rng(random_state)
    z = generate_hidden_confounder(labels, rho=rho, random_state=random_state)
    linear_acceptance = g_logits - confounder_gamma * z
    offset = _calibrate_acceptance_offset(linear_acceptance, target_acceptance=1.0 - rejection_rate)
    propensity = np.clip(_sigmoid(linear_acceptance + offset), 0.01, 0.99)

    n_rejected = int(round(len(x) * rejection_rate))
    n_rejected = max(1, min(len(x) - 1, n_rejected))
    tie_breaker = rng.normal(0.0, 1e-9, len(x))
    rejected_indices = np.argsort(propensity + tie_breaker, kind="mergesort")[:n_rejected]

    accepted_mask = np.ones(len(x), dtype=bool)
    accepted_mask[rejected_indices] = False
    rejected_mask = ~accepted_mask

    return {
        "X_acc": x.loc[accepted_mask].reset_index(drop=True),
        "y_acc": labels[accepted_mask].copy(),
        "X_rej": x.loc[rejected_mask].reset_index(drop=True),
        "y_rej_hidden": labels[rejected_mask].copy(),
        "accepted_mask": accepted_mask,
        "propensity": propensity,
        "z": z,
        "actual_rejection_rate": float(rejected_mask.mean()),
    }


def run_confounded_experiment(
    X: pd.DataFrame,
    y: np.ndarray,
    rho_values: list[float] | None = None,
    confounder_gammas: list[float] | None = None,
    rejection_rates: list[float] | None = None,
    teacher_config: dict | None = None,
    student_model_type: str = "lightgbm",
    tau_u: float = 0.5,
    pseudo_label_gamma: float = 2.0,
    lambda_distill: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run rho x gamma hidden-confounder stress tests."""
    x, labels = _validate_feature_label_pair(X, y)
    selected_rhos = rho_values or DEFAULT_RHO_VALUES
    selected_gammas = confounder_gammas or DEFAULT_CONFOUNDER_GAMMAS
    selected_rates = rejection_rates or DEFAULT_REJECTION_RATES
    _validate_grid(selected_rhos, selected_gammas, selected_rates)

    g_logits = _fit_observable_policy_logits(x, labels, random_state=random_state)
    rows: list[dict[str, object]] = []

    for combo_index, (rho, confounder_gamma, rejection_rate) in enumerate(
        product(selected_rhos, selected_gammas, selected_rates)
    ):
        combo_seed = random_state + combo_index
        data = confounded_rejection(
            x,
            labels,
            g_logits,
            rho=rho,
            confounder_gamma=confounder_gamma,
            rejection_rate=rejection_rate,
            random_state=combo_seed,
        )
        y_acc = data["y_acc"]
        y_rej_hidden = data["y_rej_hidden"]
        base_row = _base_result_row(
            rho=rho,
            confounder_gamma=confounder_gamma,
            rejection_rate=rejection_rate,
            data=data,
        )

        if not _has_both_classes(y_acc) or not _has_both_classes(y_rej_hidden):
            rows.append({**base_row, "model": "skipped", "skip_reason": "single_class_split"})
            continue

        trainer = UCRITrainer(
            teacher_config=dict(teacher_config or DEFAULT_TEACHER_CONFIG),
            student_model_type=student_model_type,
            tau_u=tau_u,
            gamma=pseudo_label_gamma,
            lambda_distill=lambda_distill,
            random_state=combo_seed,
        )
        ucri_result = trainer.run(data["X_acc"], y_acc, data["X_rej"])

        student_preds = ucri_result["student"].predict_proba(data["X_rej"])
        rows.append(
            _metric_row(
                base_row,
                "UCRI-CS",
                y_rej_hidden,
                student_preds,
                uncertainty=ucri_result["uncertainty"],
                pseudo_labels=ucri_result["pseudo_labels"],
            )
        )

        rows.append(
            _metric_row(
                base_row,
                "teacher",
                y_rej_hidden,
                ucri_result["teacher_probs"],
                uncertainty=ucri_result["uncertainty"],
                pseudo_labels=None,
            )
        )

        accepted_only = StudentModel(model_type=student_model_type, random_state=combo_seed)
        accepted_only.fit(data["X_acc"], y_acc)
        accepted_only_preds = accepted_only.predict_proba(data["X_rej"])
        rows.append(_metric_row(base_row, "accepted-only", y_rej_hidden, accepted_only_preds))

    return pd.DataFrame(rows)


def main(
    data_path: str,
    output_path: str,
    rho_values: list[float] | None = None,
    confounder_gammas: list[float] | None = None,
    rejection_rates: list[float] | None = None,
    teacher_config: dict | None = None,
    student_model_type: str = "lightgbm",
    max_rows: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Task21 confounded rejection simulation on LendingClub accepted loans."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")

    accepted = load_accepted(data_path)
    labeled = construct_default_label(label_maturity_filter(accepted)).dropna(subset=["default_label"]).copy()
    features = _make_accepted_shared_features(labeled)
    y = labeled["default_label"].astype(int).to_numpy()

    if len(features) > max_rows:
        rng = np.random.default_rng(random_state)
        sample_indices = np.sort(rng.choice(len(features), size=max_rows, replace=False))
        features = features.iloc[sample_indices].reset_index(drop=True)
        y = y[sample_indices]

    audit_features(features)
    result_frame = run_confounded_experiment(
        features,
        y,
        rho_values=rho_values,
        confounder_gammas=confounder_gammas,
        rejection_rates=rejection_rates,
        teacher_config=teacher_config,
        student_model_type=student_model_type,
        random_state=random_state,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output, index=False)

    printable = result_frame.dropna(subset=["AUROC"], how="all")
    for _, row in printable.iterrows():
        print(
            f"rho={row['rho']:.2f} | conf_gamma={row['confounder_gamma']:.2f} | "
            f"rate={row['rejection_rate']:.2f} | {row['model']}: "
            f"AUROC={row['AUROC']:.4f}, KS={row['KS']:.4f}, Brier={row['Brier']:.4f}, "
            f"unc={row['mean_uncertainty']:.4f}"
        )

    return result_frame


def _fit_observable_policy_logits(X: pd.DataFrame, y: np.ndarray, random_state: int) -> np.ndarray:
    matrix = _coerce_numeric_matrix(X)
    model = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", random_state=random_state)
    model.fit(matrix, y)
    propensity = np.clip(model.predict_proba(matrix)[:, 1], 0.01, 0.99)
    return _logit(propensity)


def _base_result_row(
    rho: float,
    confounder_gamma: float,
    rejection_rate: float,
    data: dict[str, object],
) -> dict[str, object]:
    y_acc = np.asarray(data["y_acc"], dtype=int)
    y_rej_hidden = np.asarray(data["y_rej_hidden"], dtype=int)
    z = np.asarray(data["z"], dtype=float)
    accepted_mask = np.asarray(data["accepted_mask"], dtype=bool)
    rejected_mask = ~accepted_mask
    return {
        "protocol": "ConfoundedSimulation",
        "rho": float(rho),
        "confounder_gamma": float(confounder_gamma),
        "rejection_rate": float(rejection_rate),
        "actual_rejection_rate": float(data["actual_rejection_rate"]),
        "n_accepted": int(accepted_mask.sum()),
        "n_rejected": int(rejected_mask.sum()),
        "accepted_bad_rate": float(y_acc.mean()),
        "rejected_hidden_bad_rate": float(y_rej_hidden.mean()),
        "accepted_mean_hidden_confounder": float(z[accepted_mask].mean()),
        "rejected_mean_hidden_confounder": float(z[rejected_mask].mean()),
        "skip_reason": None,
    }


def _metric_row(
    base_row: dict[str, object],
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray | None = None,
    pseudo_labels: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    metrics = compute_all_metrics(y_true, np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0))
    diagnostics = _uncertainty_diagnostics(uncertainty)
    diagnostics.update(_pseudo_label_diagnostics(pseudo_labels, y_true))
    return {
        **base_row,
        "model": model_name,
        **metrics,
        **diagnostics,
    }


def _uncertainty_diagnostics(uncertainty: np.ndarray | None) -> dict[str, float]:
    if uncertainty is None:
        return {"mean_uncertainty": 0.0, "median_uncertainty": 0.0, "high_uncertainty_rate": 0.0}
    values = np.asarray(uncertainty, dtype=float)
    median = float(np.median(values))
    return {
        "mean_uncertainty": float(values.mean()),
        "median_uncertainty": median,
        "high_uncertainty_rate": float((values > median).mean()),
    }


def _pseudo_label_diagnostics(
    pseudo_labels: dict[str, np.ndarray] | None,
    y_true: np.ndarray,
) -> dict[str, float]:
    if pseudo_labels is None:
        return {"pseudo_label_coverage": 0.0, "pseudo_label_precision": float("nan")}
    weights = np.asarray(pseudo_labels["weight"], dtype=float)
    soft_labels = np.asarray(pseudo_labels["soft_label"], dtype=float)
    selected = weights > 0
    coverage = float(selected.mean())
    if not selected.any():
        return {"pseudo_label_coverage": coverage, "pseudo_label_precision": float("nan")}
    hard_labels = (soft_labels >= 0.5).astype(int)
    return {
        "pseudo_label_coverage": coverage,
        "pseudo_label_precision": float((hard_labels[selected] == np.asarray(y_true)[selected]).mean()),
    }


def _make_accepted_shared_features(labeled: pd.DataFrame) -> pd.DataFrame:
    shared = build_shared_features(labeled, pd.DataFrame(), risk_score_setting="no_riskscore")
    shared = shared[shared["source"].eq("accepted")].drop(columns=["source", "accepted_indicator"], errors="ignore")
    if len(shared) != len(labeled):
        raise ValueError("Shared feature construction changed the accepted row count.")
    if shared.shape[1] == 0:
        raise ValueError("Confounded simulation requires at least one shared accepted feature.")
    return shared.reset_index(drop=True)


def _coerce_numeric_matrix(X: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(X).copy()
    numeric = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric[column] = pd.to_numeric(series, errors="coerce")
        elif "date" in str(column).lower() or column in {"issue_d"}:
            parsed = _parse_dates(series)
            numeric[column] = parsed.astype("int64").astype(float)
            numeric.loc[parsed.isna(), column] = np.nan
        else:
            numeric[column] = pd.Categorical(series.astype("string").fillna("missing")).codes.astype(float)
    if numeric.shape[1] == 0:
        raise ValueError("At least one feature column is required.")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    medians = numeric.median(numeric_only=True).fillna(0.0)
    return numeric.fillna(medians).astype(float)


def _parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="%Y-%m-%d")
    if parsed.notna().any():
        return parsed
    parsed = pd.to_datetime(series, errors="coerce", format="%Y-%m")
    if parsed.notna().any():
        return parsed
    return pd.to_datetime(series, errors="coerce", format="%b-%Y")


def _calibrate_acceptance_offset(linear_score: np.ndarray, target_acceptance: float) -> float:
    low, high = -30.0, 30.0
    for _ in range(80):
        mid = (low + high) / 2.0
        mean_acceptance = _sigmoid(linear_score + mid).mean()
        if mean_acceptance < target_acceptance:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values >= 0, 1.0 / (1.0 + np.exp(-values)), np.exp(values) / (1.0 + np.exp(values)))


def _logit(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.01, 0.99)
    return np.log(probabilities / (1.0 - probabilities))


def _validate_feature_label_pair(X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    x = pd.DataFrame(X).copy().reset_index(drop=True)
    labels = _validate_binary_labels(y)
    if len(x) != len(labels):
        raise ValueError("X and y must have the same length.")
    if len(x) < 2:
        raise ValueError("X and y must contain at least two rows.")
    if not x.columns.is_unique:
        raise ValueError("X feature names must be unique.")
    if not _has_both_classes(labels):
        raise ValueError("y must contain both classes for confounded simulation.")
    return x, labels


def _validate_binary_labels(y: np.ndarray) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")
    if len(labels) == 0:
        raise ValueError("y must not be empty.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y must contain binary 0/1 labels.")
    return labels.astype(int)


def _validate_g_logits(g_logits: np.ndarray, expected_length: int) -> np.ndarray:
    logits = np.asarray(g_logits, dtype=float)
    if logits.ndim != 1:
        raise ValueError("g_logits must be a one-dimensional array.")
    if len(logits) != expected_length:
        raise ValueError("g_logits must have the same length as X.")
    if not np.all(np.isfinite(logits)):
        raise ValueError("g_logits must contain finite values.")
    return logits


def _validate_rho(rho: float) -> float:
    rho = float(rho)
    if not np.isfinite(rho) or rho < 0.0 or rho > 1.0:
        raise ValueError("rho must be in [0, 1].")
    return rho


def _validate_non_negative(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite value.")
    return value


def _validate_rejection_rate(rejection_rate: float) -> float:
    rejection_rate = float(rejection_rate)
    if not np.isfinite(rejection_rate) or rejection_rate <= 0.0 or rejection_rate >= 1.0:
        raise ValueError("rejection_rate must be in (0, 1).")
    return rejection_rate


def _validate_grid(rho_values: list[float], confounder_gammas: list[float], rejection_rates: list[float]) -> None:
    if not rho_values or not confounder_gammas or not rejection_rates:
        raise ValueError("Confounded simulation grid values must not be empty.")
    for rho, confounder_gamma, rejection_rate in product(rho_values, confounder_gammas, rejection_rates):
        _validate_rho(rho)
        _validate_non_negative("confounder_gamma", confounder_gamma)
        _validate_rejection_rate(rejection_rate)


def _has_both_classes(y: np.ndarray) -> bool:
    return len(np.unique(np.asarray(y))) == 2


def _parse_float_list(value: str | None) -> list[float] | None:
    if value is None or value.strip() == "":
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run Task21 confounded rejection simulation.")
    parser.add_argument("data_path")
    parser.add_argument("output_path")
    parser.add_argument("--rho-values", default=None, help="Comma-separated rho values, e.g. 0,0.2,0.4,0.6")
    parser.add_argument("--confounder-gammas", default=None, help="Comma-separated hidden-confounder strengths.")
    parser.add_argument("--rejection-rates", default=None, help="Comma-separated rates, e.g. 0.2,0.4,0.6")
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        output_path=args.output_path,
        rho_values=_parse_float_list(args.rho_values),
        confounder_gammas=_parse_float_list(args.confounder_gammas),
        rejection_rates=_parse_float_list(args.rejection_rates),
        max_rows=args.max_rows,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    _cli()
