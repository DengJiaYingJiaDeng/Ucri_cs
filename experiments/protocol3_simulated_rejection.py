from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data.leakage_audit import audit_features
from src.data.loader import load_accepted
from src.data.preprocess import build_accepted_rich_features, construct_default_label, label_maturity_filter
from src.evaluation.metrics import compute_all_metrics
from src.models.student import StudentModel
from src.reject_inference.ssl_trainer import UCRITrainer


SIMULATION_MECHANISMS = ("logistic", "rule_based", "score_band", "geography_time", "nonlinear_rf")
OVERLAP_LEVELS = ("high", "medium", "low")
DEFAULT_TEACHER_CONFIG = {"n_models": 3, "model_types": ["lightgbm", "catboost", "mlp"]}


def simulate_rejection(
    X: pd.DataFrame,
    y: np.ndarray,
    mechanism: str = "logistic",
    rejection_rate: float = 0.4,
    overlap_level: str = "medium",
    policy_noise: float = 0.0,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Split accepted data into visible accepted loans and simulated rejected loans.

    The rejected labels are returned only for protocol evaluation and should not be
    passed to reject-inference training code.
    """
    x, labels = _validate_feature_label_pair(X, y)
    _validate_simulation_options(mechanism, rejection_rate, overlap_level, policy_noise)

    rng = np.random.default_rng(random_state)
    raw_scores = _compute_rejection_scores(x, labels, mechanism, rng, random_state)
    rejection_scores = _blend_for_overlap(raw_scores, overlap_level, rng)
    if policy_noise > 0:
        rejection_scores = (1.0 - policy_noise) * rejection_scores + policy_noise * rng.random(len(x))
    rejection_scores = _normalize_scores(rejection_scores, rng)

    n_rejected = int(round(len(x) * rejection_rate))
    n_rejected = max(1, min(len(x) - 1, n_rejected))
    rejected_indices = np.argsort(rejection_scores, kind="mergesort")[-n_rejected:]

    accepted_mask = np.ones(len(x), dtype=bool)
    accepted_mask[rejected_indices] = False
    rejected_mask = ~accepted_mask

    return (
        accepted_mask,
        x.loc[accepted_mask].reset_index(drop=True),
        labels[accepted_mask].copy(),
        x.loc[rejected_mask].reset_index(drop=True),
        labels[rejected_mask].copy(),
    )


def run_protocol_3(
    X: pd.DataFrame,
    y: np.ndarray,
    mechanisms: list[str] | None = None,
    rejection_rates: list[float] | None = None,
    overlap_levels: list[str] | None = None,
    policy_noises: list[float] | None = None,
    teacher_config: dict | None = None,
    student_model_type: str = "lightgbm",
    tau_u: float = 0.5,
    gamma: float = 2.0,
    lambda_distill: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Protocol 3 over simulated rejection settings with hidden rejected labels."""
    x, labels = _validate_feature_label_pair(X, y)
    selected_mechanisms = mechanisms or list(SIMULATION_MECHANISMS)
    selected_rates = rejection_rates or [0.2, 0.4, 0.6]
    selected_overlaps = overlap_levels or list(OVERLAP_LEVELS)
    selected_noises = policy_noises or [0.0, 0.1, 0.2]
    _validate_protocol_grid(selected_mechanisms, selected_rates, selected_overlaps, selected_noises)

    rows: list[dict[str, object]] = []
    for combo_index, (mechanism, rejection_rate, overlap_level, policy_noise) in enumerate(
        product(selected_mechanisms, selected_rates, selected_overlaps, selected_noises)
    ):
        combo_seed = random_state + combo_index
        accepted_mask, X_accepted, y_accepted, X_rejected, y_rejected_hidden = simulate_rejection(
            x,
            labels,
            mechanism=mechanism,
            rejection_rate=rejection_rate,
            overlap_level=overlap_level,
            policy_noise=policy_noise,
            random_state=combo_seed,
        )
        distribution = compute_rejection_distribution_comparison(
            X_accepted,
            X_rejected,
            random_state=combo_seed,
        )
        base_row = {
            "protocol": "Protocol3",
            "mechanism": mechanism,
            "rejection_rate": float(rejection_rate),
            "overlap_level": overlap_level,
            "policy_noise": float(policy_noise),
            "n_total": int(len(x)),
            "n_accepted": int(accepted_mask.sum()),
            "n_rejected": int((~accepted_mask).sum()),
            "accepted_bad_rate": float(y_accepted.mean()),
            "rejected_hidden_bad_rate": float(y_rejected_hidden.mean()),
            "skip_reason": None,
            **distribution,
        }

        if not _has_both_classes(y_accepted) or not _has_both_classes(y_rejected_hidden):
            rows.append({**base_row, "model": "skipped", "skip_reason": "single_class_split"})
            continue

        trainer = UCRITrainer(
            teacher_config=dict(teacher_config or DEFAULT_TEACHER_CONFIG),
            student_model_type=student_model_type,
            tau_u=tau_u,
            gamma=gamma,
            lambda_distill=lambda_distill,
            random_state=combo_seed,
        )
        ucri_result = trainer.run(X_accepted, y_accepted, X_rejected)

        ucri_predictions = ucri_result["student"].predict_proba(X_rejected)
        rows.append(_metric_row(base_row, "UCRI-CS", y_rejected_hidden, ucri_predictions))

        teacher_predictions = np.asarray(ucri_result["teacher_probs"], dtype=float)
        rows.append(_metric_row(base_row, "teacher", y_rejected_hidden, teacher_predictions))

        accepted_only = StudentModel(model_type=student_model_type, random_state=combo_seed)
        accepted_only.fit(X_accepted, y_accepted)
        accepted_only_predictions = accepted_only.predict_proba(X_rejected)
        rows.append(_metric_row(base_row, "accepted-only", y_rejected_hidden, accepted_only_predictions))

    return pd.DataFrame(rows)


def compute_rejection_distribution_comparison(
    X_accepted: pd.DataFrame,
    X_rejected: pd.DataFrame,
    max_samples: int = 1000,
    random_state: int = 42,
) -> dict[str, float]:
    """Compare accepted and simulated rejected feature distributions."""
    if len(X_accepted) == 0 or len(X_rejected) == 0:
        raise ValueError("X_accepted and X_rejected must not be empty.")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")

    accepted = _sample_rows(_coerce_numeric_matrix(X_accepted), max_samples, random_state)
    rejected = _sample_rows(_coerce_numeric_matrix(X_rejected).reindex(columns=accepted.columns), max_samples, random_state + 1)
    accepted_array = accepted.to_numpy(dtype=float)
    rejected_array = rejected.to_numpy(dtype=float)

    gamma = _median_rbf_gamma(accepted_array, rejected_array)
    mmd = float(
        _mean_rbf_kernel(accepted_array, accepted_array, gamma)
        + _mean_rbf_kernel(rejected_array, rejected_array, gamma)
        - 2.0 * _mean_rbf_kernel(accepted_array, rejected_array, gamma)
    )

    return {
        "mmd_rbf": max(0.0, mmd),
        "mean_pairwise_distance": _mean_euclidean_distance(accepted_array, rejected_array),
    }


def main(
    data_path: str,
    output_path: str,
    mechanisms: list[str] | None = None,
    rejection_rates: list[float] | None = None,
    overlap_levels: list[str] | None = None,
    policy_noises: list[float] | None = None,
    teacher_config: dict | None = None,
    student_model_type: str = "lightgbm",
    max_rows: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Protocol 3 on LendingClub accepted loans and write a metrics CSV."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")

    accepted = load_accepted(data_path)
    labeled = construct_default_label(label_maturity_filter(accepted)).dropna(subset=["default_label"]).copy()

    features = build_accepted_rich_features(labeled)
    y = labeled["default_label"].astype(int).to_numpy()
    features = features.reset_index(drop=True)
    if len(features) > max_rows:
        rng = np.random.default_rng(random_state)
        sample_indices = np.sort(rng.choice(len(features), size=max_rows, replace=False))
        features = features.iloc[sample_indices].reset_index(drop=True)
        y = y[sample_indices]

    audit_features(features)
    result_frame = run_protocol_3(
        features,
        y,
        mechanisms=mechanisms,
        rejection_rates=rejection_rates,
        overlap_levels=overlap_levels,
        policy_noises=policy_noises,
        teacher_config=teacher_config,
        student_model_type=student_model_type,
        random_state=random_state,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output, index=False)

    for _, row in result_frame.dropna(subset=["AUROC"], how="all").iterrows():
        print(
            f"{row['mechanism']} | rate={row['rejection_rate']:.2f} | "
            f"{row['overlap_level']} | noise={row['policy_noise']:.2f} | "
            f"{row['model']}: AUROC={row['AUROC']:.4f}, KS={row['KS']:.4f}, Brier={row['Brier']:.4f}"
        )

    return result_frame


def _compute_rejection_scores(
    x: pd.DataFrame,
    y: np.ndarray,
    mechanism: str,
    rng: np.random.Generator,
    random_state: int,
) -> np.ndarray:
    if mechanism == "logistic":
        model = LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state)
        matrix = _coerce_numeric_matrix(x)
        model.fit(matrix, y)
        return model.predict_proba(matrix)[:, 1]

    if mechanism == "rule_based":
        return _rule_based_scores(x)

    if mechanism == "score_band":
        return _score_band_scores(x, rng)

    if mechanism == "geography_time":
        return _geography_time_scores(x, y, rng)

    if mechanism == "nonlinear_rf":
        model = RandomForestClassifier(
            n_estimators=80,
            max_depth=8,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
        matrix = _coerce_numeric_matrix(x)
        model.fit(matrix, y)
        return model.predict_proba(matrix)[:, 1]

    raise ValueError(f"Unknown simulation mechanism: {mechanism}")


def _rule_based_scores(x: pd.DataFrame) -> np.ndarray:
    matrix = _coerce_numeric_matrix(x)
    score = np.zeros(len(matrix), dtype=float)
    weight_sum = 0.0

    positive_risk_columns = ["dti", "loan_amnt", "int_rate", "revol_util", "delinq_2yrs", "open_acc"]
    negative_risk_columns = ["fico_avg", "fico_range_low", "fico_range_high", "annual_inc", "risk_score"]
    for column in positive_risk_columns:
        if column in matrix.columns:
            score += _rank01(matrix[column].to_numpy()) * 1.0
            weight_sum += 1.0
    for column in negative_risk_columns:
        if column in matrix.columns:
            score += (1.0 - _rank01(matrix[column].to_numpy())) * 1.0
            weight_sum += 1.0

    if weight_sum == 0:
        score = matrix.rank(pct=True).mean(axis=1).to_numpy(dtype=float)
    else:
        score /= weight_sum
    return score


def _score_band_scores(x: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    matrix = _coerce_numeric_matrix(x)
    for column in ["risk_score", "fico_avg", "fico_range_low", "fico_range_high"]:
        if column in matrix.columns:
            return 1.0 - _rank01(matrix[column].to_numpy())
    if matrix.shape[1] == 0:
        return rng.random(len(x))
    return 1.0 - _rank01(matrix.iloc[:, 0].to_numpy())


def _geography_time_scores(x: pd.DataFrame, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scores = np.zeros(len(x), dtype=float)

    state_column = _first_existing_column(x, ["state", "addr_state"])
    if state_column is not None:
        state_bad_rates = pd.DataFrame({"state": x[state_column].astype(str), "y": y}).groupby("state")["y"].mean()
        state_scores = x[state_column].astype(str).map(state_bad_rates).fillna(float(np.mean(y))).to_numpy(dtype=float)
        scores += _rank01(state_scores)

    date_column = _first_existing_column(x, ["application_date", "issue_d"])
    if date_column is not None:
        parsed_dates = _parse_dates(x[date_column])
        if parsed_dates.notna().any():
            date_scores = parsed_dates.rank(pct=True).fillna(0.5).to_numpy(dtype=float)
            scores += date_scores

    if np.allclose(scores, 0.0):
        scores = _rule_based_scores(x)
    return scores + 0.05 * rng.random(len(x))


def _metric_row(base_row: dict[str, object], model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    return {
        **base_row,
        "model": model_name,
        **compute_all_metrics(y_true, np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)),
    }


def _coerce_numeric_matrix(x: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(x).copy()
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


def _sample_rows(frame: pd.DataFrame, max_samples: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= max_samples:
        return frame.reset_index(drop=True)
    return frame.sample(n=max_samples, random_state=random_state).reset_index(drop=True)


def _median_rbf_gamma(accepted_array: np.ndarray, rejected_array: np.ndarray) -> float:
    combined = np.vstack([accepted_array, rejected_array])
    if len(combined) <= 1:
        return 1.0
    if len(combined) > 300:
        combined = combined[np.linspace(0, len(combined) - 1, 300, dtype=int)]
    distances = _pairwise_squared_distances(combined, combined)
    positive_distances = distances[distances > 0]
    if len(positive_distances) == 0:
        return 1.0
    median_distance = float(np.median(positive_distances))
    return 1.0 / max(median_distance, 1e-12)


def _pairwise_squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    diff = left[:, None, :] - right[None, :, :]
    return np.square(diff).sum(axis=2)


def _mean_rbf_kernel(left: np.ndarray, right: np.ndarray, gamma: float, batch_size: int = 256) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(left), batch_size):
        squared = _pairwise_squared_distances(left[start : start + batch_size], right)
        total += float(np.exp(-gamma * squared).sum())
        count += squared.size
    return total / max(count, 1)


def _mean_euclidean_distance(left: np.ndarray, right: np.ndarray, batch_size: int = 256) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(left), batch_size):
        squared = _pairwise_squared_distances(left[start : start + batch_size], right)
        total += float(np.sqrt(squared).sum())
        count += squared.size
    return total / max(count, 1)


def _rank01(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    if len(series) <= 1 or series.nunique(dropna=True) <= 1:
        return np.zeros(len(series), dtype=float)
    return series.rank(pct=True).to_numpy(dtype=float)


def _normalize_scores(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    scores = np.nan_to_num(scores, nan=np.nanmedian(scores), posinf=np.nanmax(scores), neginf=np.nanmin(scores))
    if len(scores) <= 1 or np.allclose(scores, scores[0]):
        return rng.random(len(scores))
    ranks = pd.Series(scores + rng.normal(0.0, 1e-9, len(scores))).rank(pct=True).to_numpy(dtype=float)
    return np.clip(ranks, 0.0, 1.0)


def _blend_for_overlap(values: np.ndarray, overlap_level: str, rng: np.random.Generator) -> np.ndarray:
    normalized = _normalize_scores(values, rng)
    risk_weight = {"high": 0.45, "medium": 0.75, "low": 0.95}[overlap_level]
    return risk_weight * normalized + (1.0 - risk_weight) * rng.random(len(normalized))


def _first_existing_column(x: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in x.columns:
            return column
    return None


def _has_both_classes(y: np.ndarray) -> bool:
    return len(np.unique(np.asarray(y))) == 2


def _validate_feature_label_pair(X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    x = pd.DataFrame(X).copy().reset_index(drop=True)
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")
    if len(x) != len(labels):
        raise ValueError("X and y must have the same length.")
    if len(x) < 2:
        raise ValueError("X and y must contain at least two rows.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y must contain binary 0/1 labels.")
    if not x.columns.is_unique:
        raise ValueError("X feature names must be unique.")
    if not _has_both_classes(labels):
        raise ValueError("y must contain both classes for simulated rejection.")
    return x, labels.astype(int)


def _validate_simulation_options(
    mechanism: str,
    rejection_rate: float,
    overlap_level: str,
    policy_noise: float,
) -> None:
    if mechanism not in SIMULATION_MECHANISMS:
        raise ValueError(f"mechanism must be one of {SIMULATION_MECHANISMS}.")
    if not 0.0 < float(rejection_rate) < 1.0:
        raise ValueError("rejection_rate must be in (0, 1).")
    if overlap_level not in OVERLAP_LEVELS:
        raise ValueError(f"overlap_level must be one of {OVERLAP_LEVELS}.")
    if not 0.0 <= float(policy_noise) <= 1.0:
        raise ValueError("policy_noise must be in [0, 1].")


def _validate_protocol_grid(
    mechanisms: list[str],
    rejection_rates: list[float],
    overlap_levels: list[str],
    policy_noises: list[float],
) -> None:
    if not mechanisms or not rejection_rates or not overlap_levels or not policy_noises:
        raise ValueError("Protocol 3 grid values must not be empty.")
    for mechanism, rejection_rate, overlap_level, policy_noise in product(
        mechanisms,
        rejection_rates,
        overlap_levels,
        policy_noises,
    ):
        _validate_simulation_options(mechanism, rejection_rate, overlap_level, policy_noise)


def _parse_string_list(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_float_list(value: str | None) -> list[float] | None:
    if value is None or value.strip() == "":
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run Protocol 3 simulated rejection benchmark.")
    parser.add_argument("data_path")
    parser.add_argument("output_path")
    parser.add_argument("--mechanisms", default=None, help="Comma-separated mechanisms.")
    parser.add_argument("--rejection-rates", default=None, help="Comma-separated rates, e.g. 0.2,0.4,0.6")
    parser.add_argument("--overlap-levels", default=None, help="Comma-separated levels: high,medium,low")
    parser.add_argument("--policy-noises", default=None, help="Comma-separated noise levels, e.g. 0,0.1,0.2")
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    main(
        data_path=args.data_path,
        output_path=args.output_path,
        mechanisms=_parse_string_list(args.mechanisms),
        rejection_rates=_parse_float_list(args.rejection_rates),
        overlap_levels=_parse_string_list(args.overlap_levels),
        policy_noises=_parse_float_list(args.policy_noises),
        max_rows=args.max_rows,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    _cli()
