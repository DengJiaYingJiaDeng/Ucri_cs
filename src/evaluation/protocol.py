from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.baselines.traditional import TRADITIONAL_BASELINES
from src.decision.profit import (
    compute_expected_profit,
    compute_oracle_profit,
    compute_oracle_profit_ratio,
    compute_random_profit,
)
from src.decision.threshold import DecisionThresholdOptimizer, DecisionThresholds
from src.evaluation.metrics import compute_all_metrics, compute_brier, compute_ece, compute_ece_equal_width, compute_psi
from src.models.device import validate_device_type, validate_gpu_device_id


PROTOCOL4_PERIOD_TYPES = {
    "validation": "validation",
    "test_normal": "normal_drift",
    "test_extended": "extended_drift",
    "test_structural_break": "structural_break_stress",
}
PROTOCOL4_MAIN_PERIODS = ("test_normal", "test_extended")
PROTOCOL5_TARGET_BAD_RATES = (0.05, 0.08, 0.10, 0.12)
PROTOCOL5_MIN_APPROVAL_RATES = (0.20, 0.30, 0.40, 0.50)
PROTOCOL5_LGD_VALUES = (0.20, 0.35, 0.45, 0.60, 0.75, 0.90)
PROTOCOL6_DEFAULT_GROUP_COLUMNS = (
    "state",
    "zip3_region",
    "loan_purpose",
    "employment_length_group",
    "income_group",
    "risk_score_band",
)


@dataclass
class ProtocolResult:
    protocol: str
    model_name: str
    metrics: dict[str, float]
    predictions: np.ndarray
    true_labels: np.ndarray


def run_protocol_1(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_names: list[str] | None = None,
    random_state: int = 42,
) -> list[ProtocolResult]:
    """Run Protocol 1: accepted-only out-of-time PD benchmark."""
    x_train, y_train = _validate_feature_label_pair("train", X_train, y_train)
    _validate_feature_label_pair("validation", X_val, y_val)
    x_test, y_test = _validate_feature_label_pair("test", X_test, y_test)

    x_test = x_test.reindex(columns=x_train.columns)
    selected_models = model_names or ["LogisticRegression", "LightGBM", "CatBoost"]
    _validate_model_names(selected_models)

    results = []
    for name in selected_models:
        model = TRADITIONAL_BASELINES[name](random_state=random_state)
        model.fit(x_train, y_train)
        predictions = _predict_positive_probability(model, x_test)
        metrics = compute_all_metrics(y_test, predictions)
        results.append(
            ProtocolResult(
                protocol="Protocol1",
                model_name=name,
                metrics=metrics,
                predictions=predictions,
                true_labels=y_test.copy(),
            )
        )
    return results


def run_protocol_4(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    periods: dict[str, tuple[pd.DataFrame, np.ndarray]],
    model_names: list[str] | None = None,
    approval_pd_threshold: float = 0.5,
    device_type: str = "cpu",
    gpu_device_id: int = 0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Protocol 4: policy-shift and temporal-stability evaluation."""
    x_train, y_train = _validate_feature_label_pair("train", X_train, y_train)
    _validate_protocol4_periods(periods)
    if not 0.0 <= float(approval_pd_threshold) <= 1.0:
        raise ValueError("approval_pd_threshold must be in [0, 1].")
    validated_device = validate_device_type(device_type)
    validated_gpu_device_id = validate_gpu_device_id(gpu_device_id)

    selected_models = model_names or ["LogisticRegression", "LightGBM", "CatBoost"]
    _validate_model_names(selected_models)

    rows: list[dict[str, object]] = []
    for name in selected_models:
        model = _build_protocol_model(name, random_state, validated_device, validated_gpu_device_id)
        model.fit(x_train, y_train)
        train_predictions = _predict_positive_probability(model, x_train)

        model_rows: list[dict[str, object]] = []
        for period_name in _ordered_protocol4_periods(periods):
            X_period, y_period = periods[period_name]
            row_base = _protocol4_base_row(
                model_name=name,
                period_name=period_name,
                n_train=len(x_train),
                approval_pd_threshold=approval_pd_threshold,
            )
            if len(pd.DataFrame(X_period)) == 0:
                model_rows.append({**row_base, "n_samples": 0, "skip_reason": "empty_period"})
                continue

            x_period, labels = _validate_feature_label_pair(period_name, X_period, y_period)
            x_period = x_period.reindex(columns=x_train.columns)
            predictions = _predict_positive_probability(model, x_period)
            metrics = _compute_protocol4_metrics(labels, predictions)
            approval = _compute_approval_summary(labels, predictions, approval_pd_threshold)

            model_rows.append(
                {
                    **row_base,
                    "n_samples": int(len(x_period)),
                    "skip_reason": None,
                    "score_mean": float(np.mean(predictions)),
                    "score_std": float(np.std(predictions)),
                    "score_psi_vs_train": compute_psi(train_predictions, predictions),
                    **approval,
                    **metrics,
                }
            )

        rows.extend(_add_protocol4_drift_columns(model_rows))

    return pd.DataFrame(rows)


def run_protocol_5(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    loan_amounts_test: np.ndarray,
    model_names: list[str] | None = None,
    target_bad_rates: list[float] | None = None,
    min_approval_rates: list[float] | None = None,
    lgd_values: list[float] | None = None,
    interest_rate: float = 0.10,
    funding_cost: float = 0.04,
    servicing_cost: float = 0.0,
    prepayment_haircut: float = 1.0,
    term_years: float = 3.0,
    device_type: str = "cpu",
    gpu_device_id: int = 0,
    random_state: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run Protocol 5: decision-aware approval simulation."""
    x_train, y_train = _validate_feature_label_pair("train", X_train, y_train)
    x_validation, y_validation = _validate_feature_label_pair("validation", X_validation, y_validation)
    x_test, y_test = _validate_feature_label_pair("test", X_test, y_test)
    x_validation = x_validation.reindex(columns=x_train.columns)
    x_test = x_test.reindex(columns=x_train.columns)
    loan_amounts_test = _validate_loan_amounts(loan_amounts_test, expected_length=len(x_test))

    selected_models = model_names or ["LogisticRegression", "LightGBM", "CatBoost"]
    _validate_model_names(selected_models)
    target_bad_rates = _validate_probability_grid(
        "target_bad_rates",
        target_bad_rates or list(PROTOCOL5_TARGET_BAD_RATES),
    )
    min_approval_rates = _validate_probability_grid(
        "min_approval_rates",
        min_approval_rates or list(PROTOCOL5_MIN_APPROVAL_RATES),
    )
    lgd_values = _validate_probability_grid("lgd_values", lgd_values or list(PROTOCOL5_LGD_VALUES))
    validated_device = validate_device_type(device_type)
    validated_gpu_device_id = validate_gpu_device_id(gpu_device_id)

    rows: list[dict[str, object]] = []
    for model_name in selected_models:
        if verbose:
            print(f"Protocol 5 fitting {model_name}...", flush=True)
        model = _build_protocol_model(model_name, random_state, validated_device, validated_gpu_device_id)
        model.fit(x_train, y_train)
        if verbose:
            print(f"Protocol 5 predicting {model_name}...", flush=True)
        validation_predictions = _predict_positive_probability(model, x_validation)
        test_predictions = _predict_positive_probability(model, x_test)
        model_metrics = compute_all_metrics(y_test, test_predictions)
        if verbose:
            print(f"Protocol 5 optimizing decision grid for {model_name}...", flush=True)

        for target_bad_rate in target_bad_rates:
            for min_approval_rate in min_approval_rates:
                optimizer = DecisionThresholdOptimizer(
                    target_bad_rate=target_bad_rate,
                    min_approval_rate=min_approval_rate,
                )
                thresholds = optimizer.optimize(y_validation, validation_predictions)
                validation_decisions = optimizer.apply(validation_predictions, thresholds)
                test_decisions = optimizer.apply(test_predictions, thresholds)
                approved = test_decisions == "approve"
                validation_approved = validation_decisions == "approve"
                validation_approval_rate = float(validation_approved.mean())
                validation_realized_bad_rate = _safe_bad_rate(y_validation, validation_approved)
                validation_constraint_feasible = (
                    validation_approval_rate >= float(min_approval_rate)
                    and validation_realized_bad_rate <= float(target_bad_rate)
                )
                base_row = {
                    "protocol": "Protocol5",
                    "model": model_name,
                    "evaluation_population": "future_accepted_test",
                    "n_train": int(len(x_train)),
                    "n_validation": int(len(x_validation)),
                    "n_test": int(len(x_test)),
                    "target_bad_rate": float(target_bad_rate),
                    "min_approval_rate": float(min_approval_rate),
                    "theta_approve": thresholds.theta_approve,
                    "theta_reject": thresholds.theta_reject,
                    "validation_approval_rate": validation_approval_rate,
                    "validation_realized_bad_rate": validation_realized_bad_rate,
                    "validation_constraint_feasible": bool(validation_constraint_feasible),
                    "approval_rate": float(approved.mean()),
                    "reject_rate": float((test_decisions == "reject").mean()),
                    "manual_review_rate": float((test_decisions == "manual_review").mean()),
                    "realized_bad_rate": _safe_bad_rate(y_test, approved),
                    "average_calibrated_pd": float(np.mean(test_predictions)),
                    "approved_average_calibrated_pd": _safe_prediction_mean(test_predictions, approved),
                    "ks_at_approval_boundary": _ks_at_approval_boundary(y_test, test_predictions, thresholds.theta_approve),
                    "AUROC": model_metrics["AUROC"],
                    "PR-AUC": model_metrics["PR-AUC"],
                    "KS": model_metrics["KS"],
                    "Brier": model_metrics["Brier"],
                    "ECE": model_metrics["ECE"],
                }
                for lgd in lgd_values:
                    rows.append(
                        {
                            **base_row,
                            "lgd": float(lgd),
                            **_protocol5_profit_summary(
                                y_true=y_test,
                                y_pred=test_predictions,
                                approved=approved,
                                loan_amounts=loan_amounts_test,
                                lgd=lgd,
                                interest_rate=interest_rate,
                                funding_cost=funding_cost,
                                servicing_cost=servicing_cost,
                                prepayment_haircut=prepayment_haircut,
                                term_years=term_years,
                                random_state=random_state,
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def run_protocol_6(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    group_features_test: pd.DataFrame,
    loan_amounts_test: np.ndarray,
    model_names: list[str] | None = None,
    group_columns: list[str] | None = None,
    target_bad_rate: float = 0.08,
    min_approval_rate: float = 0.20,
    lgd: float = 0.45,
    approval_pd_threshold: float | None = None,
    min_group_size: int = 20,
    interest_rate: float = 0.10,
    funding_cost: float = 0.04,
    servicing_cost: float = 0.0,
    prepayment_haircut: float = 1.0,
    term_years: float = 3.0,
    device_type: str = "cpu",
    gpu_device_id: int = 0,
    random_state: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run Protocol 6: subgroup robustness and fairness audit."""
    x_train, y_train = _validate_feature_label_pair("train", X_train, y_train)
    x_validation, y_validation = _validate_feature_label_pair("validation", X_validation, y_validation)
    x_test, y_test = _validate_feature_label_pair("test", X_test, y_test)
    x_validation = x_validation.reindex(columns=x_train.columns)
    x_test = x_test.reindex(columns=x_train.columns)
    groups = _validate_group_features(group_features_test, expected_length=len(x_test))
    loan_amounts_test = _validate_loan_amounts(loan_amounts_test, expected_length=len(x_test))

    selected_models = model_names or ["LogisticRegression", "LightGBM", "CatBoost"]
    _validate_model_names(selected_models)
    selected_group_columns = _select_protocol6_group_columns(groups, group_columns)
    target_bad_rate = _validate_probability_value("target_bad_rate", target_bad_rate)
    min_approval_rate = _validate_probability_value("min_approval_rate", min_approval_rate)
    lgd = _validate_probability_value("lgd", lgd)
    if approval_pd_threshold is not None:
        approval_pd_threshold = _validate_probability_value("approval_pd_threshold", approval_pd_threshold)
    min_group_size = _validate_min_group_size(min_group_size)
    validated_device = validate_device_type(device_type)
    validated_gpu_device_id = validate_gpu_device_id(gpu_device_id)

    rows: list[dict[str, object]] = []
    for model_name in selected_models:
        if verbose:
            print(f"Protocol 6 fitting {model_name}...", flush=True)
        model = _build_protocol_model(model_name, random_state, validated_device, validated_gpu_device_id)
        model.fit(x_train, y_train)
        validation_predictions = _predict_positive_probability(model, x_validation)
        test_predictions = _predict_positive_probability(model, x_test)
        thresholds = _protocol6_thresholds(
            y_validation,
            validation_predictions,
            target_bad_rate=target_bad_rate,
            min_approval_rate=min_approval_rate,
            approval_pd_threshold=approval_pd_threshold,
        )
        decisions = DecisionThresholdOptimizer().apply(test_predictions, thresholds)
        overall_metrics = _compute_protocol6_metrics(
            y_test,
            test_predictions,
            decisions,
            loan_amounts_test,
            lgd=lgd,
            interest_rate=interest_rate,
            funding_cost=funding_cost,
            servicing_cost=servicing_cost,
            prepayment_haircut=prepayment_haircut,
            term_years=term_years,
        )
        base_row = {
            "protocol": "Protocol6",
            "model": model_name,
            "evaluation_population": "future_accepted_test",
            "n_train": int(len(x_train)),
            "n_validation": int(len(x_validation)),
            "n_test": int(len(x_test)),
            "target_bad_rate": target_bad_rate,
            "min_approval_rate": min_approval_rate,
            "lgd": lgd,
            "theta_approve": thresholds.theta_approve,
            "theta_reject": thresholds.theta_reject,
            "overall_approval_rate": overall_metrics["approval_rate"],
            "overall_approved_bad_rate": overall_metrics["approved_bad_rate"],
            "overall_equal_opportunity_rate": overall_metrics["equal_opportunity_rate"],
            "overall_manual_review_rate": overall_metrics["manual_review_rate"],
            "overall_profit_per_loan": overall_metrics["profit_per_loan"],
            "overall_expected_profit": overall_metrics["expected_profit"],
            "overall_AUROC": overall_metrics["AUROC"],
            "overall_Brier": overall_metrics["Brier"],
            "overall_ECE": overall_metrics["ECE"],
            "overall_default_rate": overall_metrics["default_rate"],
        }

        for group_column in selected_group_columns:
            group_rows = _protocol6_group_rows(
                groups=groups,
                group_column=group_column,
                y_true=y_test,
                y_pred=test_predictions,
                decisions=decisions,
                loan_amounts=loan_amounts_test,
                min_group_size=min_group_size,
                metric_kwargs={
                    "lgd": lgd,
                    "interest_rate": interest_rate,
                    "funding_cost": funding_cost,
                    "servicing_cost": servicing_cost,
                    "prepayment_haircut": prepayment_haircut,
                    "term_years": term_years,
                },
            )
            disparity = _protocol6_disparity_summary(group_rows)
            for row in group_rows:
                rows.append({**base_row, **row, **disparity})

    return pd.DataFrame(rows)


def _validate_feature_label_pair(name: str, X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    x = pd.DataFrame(X).copy().reset_index(drop=True)
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError(f"{name} labels must be a one-dimensional array.")
    if len(x) != len(labels):
        raise ValueError(f"{name} features and labels must have the same length.")
    if len(x) == 0:
        raise ValueError(f"{name} features and labels must not be empty.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{name} labels must be binary 0/1 values.")
    if not x.columns.is_unique:
        raise ValueError(f"{name} features must have unique column names.")
    return x, labels.astype(int)


def _validate_protocol4_periods(periods: dict[str, tuple[pd.DataFrame, np.ndarray]]) -> None:
    if not periods:
        raise ValueError("Protocol 4 periods must include validation and must not be empty.")
    if "validation" not in periods:
        raise ValueError("Protocol 4 periods must include a non-empty validation period.")
    X_validation, y_validation = periods["validation"]
    if len(pd.DataFrame(X_validation)) == 0 or len(np.asarray(y_validation)) == 0:
        raise ValueError("Protocol 4 validation period must not be empty.")


def _protocol5_profit_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    approved: np.ndarray,
    loan_amounts: np.ndarray,
    lgd: float,
    interest_rate: float,
    funding_cost: float,
    servicing_cost: float,
    prepayment_haircut: float,
    term_years: float,
    random_state: int,
) -> dict[str, float]:
    profit_kwargs = {
        "lgd": lgd,
        "interest_rate": interest_rate,
        "funding_cost": funding_cost,
        "servicing_cost": servicing_cost,
        "prepayment_haircut": prepayment_haircut,
        "term_years": term_years,
    }
    model_profit = compute_expected_profit(y_pred, y_true, approved, loan_amounts, **profit_kwargs)
    oracle_profit = compute_oracle_profit(y_pred, y_true, loan_amounts, **profit_kwargs)
    random_profit = compute_random_profit(
        y_true,
        loan_amounts,
        approval_rate=float(approved.mean()),
        random_state=random_state,
        **profit_kwargs,
    )
    return {
        "expected_profit": model_profit["total_profit"],
        "profit_per_loan": model_profit["profit_per_loan"],
        "oracle_profit": oracle_profit["total_profit"],
        "random_profit": random_profit["total_profit"],
        "oracle_profit_gap": float(oracle_profit["total_profit"] - model_profit["total_profit"]),
        "oracle_profit_ratio": compute_oracle_profit_ratio(
            model_profit["total_profit"],
            oracle_profit["total_profit"],
            random_profit["total_profit"],
        ),
    }


def _validate_loan_amounts(values: np.ndarray, expected_length: int) -> np.ndarray:
    amounts = np.asarray(values, dtype=float)
    if amounts.ndim != 1:
        raise ValueError("loan_amounts_test must be a one-dimensional array.")
    if len(amounts) != expected_length:
        raise ValueError("loan_amounts_test and X_test must have the same length.")
    if len(amounts) == 0:
        raise ValueError("loan_amounts_test must not be empty.")
    if not np.all(np.isfinite(amounts)) or np.any(amounts < 0):
        raise ValueError("loan_amounts_test must contain non-negative finite values.")
    return amounts


def _validate_group_features(group_features: pd.DataFrame, expected_length: int) -> pd.DataFrame:
    groups = pd.DataFrame(group_features).copy().reset_index(drop=True)
    if len(groups) != expected_length:
        raise ValueError("group_features_test and X_test must have the same length.")
    if len(groups) == 0:
        raise ValueError("group_features_test must not be empty.")
    if not groups.columns.is_unique:
        raise ValueError("group_features_test must have unique column names.")
    return groups


def _select_protocol6_group_columns(groups: pd.DataFrame, group_columns: list[str] | None) -> list[str]:
    if group_columns is not None:
        if len(group_columns) == 0:
            raise ValueError("group_columns must not be empty.")
        missing = [column for column in group_columns if column not in groups.columns]
        if missing:
            raise KeyError(f"Unknown Protocol 6 group column(s): {missing}")
        return group_columns

    selected = [
        column
        for column in PROTOCOL6_DEFAULT_GROUP_COLUMNS
        if column in groups.columns and groups[column].notna().any()
    ]
    if not selected:
        raise ValueError("Protocol 6 requires at least one subgroup column.")
    return selected


def _protocol6_thresholds(
    y_validation: np.ndarray,
    validation_predictions: np.ndarray,
    target_bad_rate: float,
    min_approval_rate: float,
    approval_pd_threshold: float | None,
) -> DecisionThresholds:
    if approval_pd_threshold is not None:
        theta_reject = max(float(approval_pd_threshold), float(np.quantile(validation_predictions, 0.75)))
        return DecisionThresholds(
            theta_approve=float(approval_pd_threshold),
            theta_reject=float(np.clip(theta_reject, approval_pd_threshold, 1.0)),
        )

    optimizer = DecisionThresholdOptimizer(
        target_bad_rate=target_bad_rate,
        min_approval_rate=min_approval_rate,
    )
    return optimizer.optimize(y_validation, validation_predictions)


def _protocol6_group_rows(
    groups: pd.DataFrame,
    group_column: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decisions: np.ndarray,
    loan_amounts: np.ndarray,
    min_group_size: int,
    metric_kwargs: dict[str, float],
) -> list[dict[str, object]]:
    values = groups[group_column].astype(object).where(groups[group_column].notna(), "missing")
    rows: list[dict[str, object]] = []
    for group_value in sorted(values.unique(), key=lambda item: str(item)):
        mask = (values == group_value).to_numpy()
        n_group = int(mask.sum())
        row_base = {
            "group_feature": group_column,
            "group_value": str(group_value),
            "n_group": n_group,
            "group_share": float(n_group / len(values)),
        }
        if n_group < min_group_size:
            rows.append(
                {
                    **row_base,
                    "skip_reason": "below_min_group_size",
                    **_empty_protocol6_metrics(),
                }
            )
            continue

        rows.append(
            {
                **row_base,
                "skip_reason": None,
                **_compute_protocol6_metrics(
                    y_true[mask],
                    y_pred[mask],
                    decisions[mask],
                    loan_amounts[mask],
                    **metric_kwargs,
                ),
            }
        )
    return rows


def _compute_protocol6_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decisions: np.ndarray,
    loan_amounts: np.ndarray,
    lgd: float,
    interest_rate: float,
    funding_cost: float,
    servicing_cost: float,
    prepayment_haircut: float,
    term_years: float,
) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    predictions = np.asarray(y_pred, dtype=float)
    decisions = np.asarray(decisions, dtype=object)
    loan_amounts = np.asarray(loan_amounts, dtype=float)
    approved = decisions == "approve"
    manual_review = decisions == "manual_review"
    rejected = decisions == "reject"
    good = labels == 0
    bad = labels == 1

    metrics = _compute_protocol4_metrics(labels, predictions)
    profit = compute_expected_profit(
        predictions,
        labels,
        approved,
        loan_amounts,
        lgd=lgd,
        interest_rate=interest_rate,
        funding_cost=funding_cost,
        servicing_cost=servicing_cost,
        prepayment_haircut=prepayment_haircut,
        term_years=term_years,
    )

    return {
        "AUROC": metrics["AUROC"],
        "PR-AUC": metrics["PR-AUC"],
        "KS": metrics["KS"],
        "Brier": metrics["Brier"],
        "ECE": metrics["ECE"],
        "default_rate": metrics["default_rate"],
        "approval_rate": float(approved.mean()),
        "reject_rate": float(rejected.mean()),
        "manual_review_rate": float(manual_review.mean()),
        "approved_bad_rate": _safe_bad_rate(labels, approved),
        "equal_opportunity_rate": _safe_mask_rate(approved, good),
        "false_negative_rate": _safe_mask_rate(approved, bad),
        "false_positive_rate": _safe_mask_rate(~approved, good),
        "expected_profit": profit["total_profit"],
        "profit_per_loan": profit["profit_per_loan"],
        "profit_per_application": float(profit["total_profit"] / len(labels)),
    }


def _empty_protocol6_metrics() -> dict[str, float]:
    return {
        "AUROC": float("nan"),
        "PR-AUC": float("nan"),
        "KS": float("nan"),
        "Brier": float("nan"),
        "ECE": float("nan"),
        "default_rate": float("nan"),
        "approval_rate": float("nan"),
        "reject_rate": float("nan"),
        "manual_review_rate": float("nan"),
        "approved_bad_rate": float("nan"),
        "equal_opportunity_rate": float("nan"),
        "false_negative_rate": float("nan"),
        "false_positive_rate": float("nan"),
        "expected_profit": float("nan"),
        "profit_per_loan": float("nan"),
        "profit_per_application": float("nan"),
    }


def _protocol6_disparity_summary(group_rows: list[dict[str, object]]) -> dict[str, float]:
    evaluated_rows = [row for row in group_rows if row.get("skip_reason") is None]
    return {
        "n_groups_total": float(len(group_rows)),
        "n_groups_evaluated": float(len(evaluated_rows)),
        "AUROC_gap": _finite_gap(row.get("AUROC") for row in evaluated_rows),
        "Brier_gap": _finite_gap(row.get("Brier") for row in evaluated_rows),
        "ECE_gap": _finite_gap(row.get("ECE") for row in evaluated_rows),
        "approval_rate_gap": _finite_gap(row.get("approval_rate") for row in evaluated_rows),
        "bad_rate_gap": _finite_gap(row.get("approved_bad_rate") for row in evaluated_rows),
        "group_default_rate_gap": _finite_gap(row.get("default_rate") for row in evaluated_rows),
        "equal_opportunity_gap": _finite_gap(row.get("equal_opportunity_rate") for row in evaluated_rows),
        "false_negative_rate_gap": _finite_gap(row.get("false_negative_rate") for row in evaluated_rows),
        "false_positive_rate_gap": _finite_gap(row.get("false_positive_rate") for row in evaluated_rows),
        "manual_review_burden_gap": _finite_gap(row.get("manual_review_rate") for row in evaluated_rows),
        "profit_gap": _finite_gap(row.get("profit_per_application") for row in evaluated_rows),
        "profit_per_loan_gap": _finite_gap(row.get("profit_per_loan") for row in evaluated_rows),
    }


def _safe_mask_rate(numerator_mask: np.ndarray, denominator_mask: np.ndarray) -> float:
    numerator_mask = np.asarray(numerator_mask, dtype=bool)
    denominator_mask = np.asarray(denominator_mask, dtype=bool)
    if not denominator_mask.any():
        return float("nan")
    return float(numerator_mask[denominator_mask].mean())


def _finite_gap(values) -> float:
    finite_values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(max(finite_values) - min(finite_values))


def _validate_probability_grid(name: str, values: list[float]) -> list[float]:
    if not values:
        raise ValueError(f"{name} must not be empty.")
    result = []
    for value in values:
        value = float(value)
        if not np.isfinite(value) or value < 0 or value > 1:
            raise ValueError(f"{name} values must be in [0, 1].")
        result.append(value)
    return result


def _validate_probability_value(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"{name} must be in [0, 1].")
    return value


def _validate_min_group_size(value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError("min_group_size must be positive.")
    return value


def _safe_bad_rate(y_true: np.ndarray, mask: np.ndarray) -> float:
    labels = np.asarray(y_true)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    return float(labels[mask].mean())


def _safe_prediction_mean(y_pred: np.ndarray, mask: np.ndarray) -> float:
    predictions = np.asarray(y_pred, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    return float(predictions[mask].mean())


def _ks_at_approval_boundary(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    labels = np.asarray(y_true)
    predictions = np.asarray(y_pred, dtype=float)
    good = labels == 0
    bad = labels == 1
    if not good.any() or not bad.any():
        return float("nan")
    approved = predictions <= float(threshold)
    good_approval_rate = float(approved[good].mean())
    bad_approval_rate = float(approved[bad].mean())
    return float(abs(good_approval_rate - bad_approval_rate))


def _build_protocol_model(
    name: str,
    random_state: int,
    device_type: str,
    gpu_device_id: int,
):
    if name in {"LightGBM", "CatBoost"}:
        return TRADITIONAL_BASELINES[name](
            random_state=random_state,
            device_type=device_type,
            gpu_device_id=gpu_device_id,
        )
    return TRADITIONAL_BASELINES[name](random_state=random_state)


def _ordered_protocol4_periods(periods: dict[str, tuple[pd.DataFrame, np.ndarray]]) -> list[str]:
    ordered = [name for name in PROTOCOL4_PERIOD_TYPES if name in periods]
    extras = sorted(name for name in periods if name not in PROTOCOL4_PERIOD_TYPES)
    return ordered + extras


def _protocol4_base_row(
    model_name: str,
    period_name: str,
    n_train: int,
    approval_pd_threshold: float,
) -> dict[str, object]:
    period_type = PROTOCOL4_PERIOD_TYPES.get(period_name, "custom")
    return {
        "protocol": "Protocol4",
        "model": model_name,
        "period": period_name,
        "period_type": period_type,
        "is_structural_break_stress_test": period_name == "test_structural_break",
        "n_train": int(n_train),
        "approval_pd_threshold": float(approval_pd_threshold),
    }


def _compute_protocol4_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    labels = np.asarray(y_true)
    predictions = np.asarray(y_pred, dtype=float)
    if len(np.unique(labels)) == 2:
        return compute_all_metrics(labels, predictions)

    default_rate = float(labels.mean())
    return {
        "AUROC": float("nan"),
        "PR-AUC": float("nan"),
        "PR-AUC_baseline": default_rate,
        "default_rate": default_rate,
        "KS": float("nan"),
        "KS_threshold": float("nan"),
        "Brier": compute_brier(labels, predictions),
        "ECE": compute_ece(labels, predictions),
        "ECE_equal_width_10": compute_ece_equal_width(labels, predictions, n_bins=10),
        "ECE_equal_width_20": compute_ece_equal_width(labels, predictions, n_bins=20),
        "calib_slope": float("nan"),
        "calib_intercept": float("nan"),
    }


def _compute_approval_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    approval_pd_threshold: float,
) -> dict[str, float]:
    labels = np.asarray(y_true)
    predictions = np.asarray(y_pred, dtype=float)
    approved = predictions <= float(approval_pd_threshold)
    approved_bad_rate = float(labels[approved].mean()) if approved.any() else float("nan")
    return {
        "approval_rate": float(approved.mean()),
        "approved_bad_rate": approved_bad_rate,
    }


def _add_protocol4_drift_columns(model_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    validation_rows = [row for row in model_rows if row["period"] == "validation" and row.get("skip_reason") is None]
    if not validation_rows:
        raise ValueError("Protocol 4 requires a non-empty validation row for drift baselines.")
    validation = validation_rows[0]
    evaluated = [row for row in model_rows if row.get("skip_reason") is None]

    worst_period_auc = _finite_min(row.get("AUROC") for row in evaluated if row["period"] != "validation")
    worst_main_period_auc = _finite_min(
        row.get("AUROC")
        for row in evaluated
        if row["period"] in PROTOCOL4_MAIN_PERIODS
    )

    enriched = []
    for row in model_rows:
        enriched_row = row.copy()
        if row.get("skip_reason") is None:
            enriched_row["Brier_drift_vs_validation"] = float(row["Brier"] - validation["Brier"])
            enriched_row["ECE_drift_vs_validation"] = float(row["ECE"] - validation["ECE"])
            enriched_row["approval_rate_drift_vs_validation"] = float(
                row["approval_rate"] - validation["approval_rate"]
            )
            enriched_row["bad_rate_drift_vs_validation"] = float(row["default_rate"] - validation["default_rate"])
        else:
            enriched_row["Brier_drift_vs_validation"] = float("nan")
            enriched_row["ECE_drift_vs_validation"] = float("nan")
            enriched_row["approval_rate_drift_vs_validation"] = float("nan")
            enriched_row["bad_rate_drift_vs_validation"] = float("nan")
        enriched_row["worst_period_AUROC"] = worst_period_auc
        enriched_row["worst_main_period_AUROC"] = worst_main_period_auc
        enriched.append(enriched_row)
    return enriched


def _finite_min(values) -> float:
    finite_values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(min(finite_values))


def _validate_model_names(model_names: list[str]) -> None:
    if len(model_names) == 0:
        raise ValueError("model_names must not be empty.")
    missing = [name for name in model_names if name not in TRADITIONAL_BASELINES]
    if missing:
        raise KeyError(f"Unknown model name(s): {missing}")


def _predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.ndim == 2:
        if probabilities.shape[1] < 2:
            raise ValueError("predict_proba must return a positive-class probability column.")
        probabilities = probabilities[:, 1]
    if probabilities.ndim != 1:
        raise ValueError("predictions must be a one-dimensional array.")
    if len(probabilities) != len(X):
        raise ValueError("predictions must have the same length as X.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("predictions must contain finite values.")
    return np.clip(probabilities, 0.0, 1.0)
