"""Leakage-safe classification case-study pipeline with validation tuning.

The primary experiment uses a stratified 70/15/15 train/validation/test split.
Only validation data is used for early stopping and probability-threshold
selection; the test split is reserved for one final unbiased report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from classification.evaluation.manual_metrics import evaluate_classification
from classification.evaluation.manual_roc_auc import calculate_binary_roc_curve
from classification.data import (
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    load_machine_failure_dataset,
)
from classification.lightgbm_classification import LightGBMClassification
from lightgbm_from_scratch.metrics import average_precision, binary_precision_recall_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "case_study" / "artifacts" / "classification"


def split_classification_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    """Stratified raw-row split before model selection."""
    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("train/validation/test size phải có tổng bằng 1.")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=validation_size + test_size,
        random_state=random_state,
        stratify=y,
    )
    test_fraction = test_size / (validation_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_fraction,
        random_state=random_state,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def binary_metrics(y_true, positive_probability, threshold):
    """Calculate classification, ROC and PR metrics without sklearn metrics."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probability = np.asarray(positive_probability, dtype=float).reshape(-1)
    prediction = (probability >= float(threshold)).astype(int)
    label_metrics = evaluate_classification(
        y_true,
        prediction,
        classes=[0, 1],
        class_names=["No failure", "Machine failure"],
        positive_label=1,
    )
    matrix = label_metrics["confusion_matrix"]
    tn, fp = int(matrix[0, 0]), int(matrix[0, 1])
    fn, tp = int(matrix[1, 0]), int(matrix[1, 1])
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    roc = calculate_binary_roc_curve(y_true, probability, positive_label=1)
    ap = average_precision(y_true, probability)
    return {
        "threshold": float(threshold),
        "accuracy": float(label_metrics["accuracy"]),
        "precision": float(label_metrics["precision"]),
        "recall": float(label_metrics["recall"]),
        "f1": float(label_metrics["f1_score"]),
        "specificity": float(specificity),
        "balanced_accuracy": float((label_metrics["recall"] + specificity) / 2.0),
        "roc_auc": float(roc["auc"]),
        "pr_auc_average_precision": float(ap),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def tune_threshold(y_true, probability, thresholds=None):
    """Choose the validation threshold with max F1, then recall, then precision."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    rows = [binary_metrics(y_true, probability, threshold) for threshold in thresholds]
    frame = pd.DataFrame(rows)
    order = frame.sort_values(
        ["f1", "recall", "precision", "threshold"],
        ascending=[False, False, False, True],
    )
    best = order.iloc[0]
    return float(best["threshold"]), frame


def run_classification_case_study(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_params: dict[str, Any] | None = None,
    random_state: int = 42,
):
    """Run the full validation-driven machine-failure case study."""
    X, y = load_machine_failure_dataset(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_classification_dataset(
        X, y, random_state=random_state
    )

    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 5,
        "max_bins": 63,
        "min_child_samples": 20,
        "reg_lambda": 1.0,
        "top_rate": 0.2,
        "other_rate": 0.1,
        "feature_fraction": 1.0,
        "random_state": random_state,
        "early_stopping_rounds": 25,
    }
    if model_params:
        params.update(model_params)
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 25))

    model = LightGBMClassification(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=early_stopping_rounds,
    )

    validation_probability = model.predict_proba(X_val)[:, 1]
    chosen_threshold, threshold_frame = tune_threshold(
        y_val.to_numpy(), validation_probability
    )
    model.threshold = chosen_threshold

    split_data = {
        "train": (X_train, y_train),
        "validation": (X_val, y_val),
        "test": (X_test, y_test),
    }
    probabilities = {}
    metrics = {}
    default_threshold_metrics = {}
    prediction_frames = []
    for split_name, (features, target) in split_data.items():
        probability = model.predict_proba(features)[:, 1]
        probabilities[split_name] = probability
        metrics[split_name] = binary_metrics(
            target.to_numpy(), probability, chosen_threshold
        )
        default_threshold_metrics[split_name] = binary_metrics(
            target.to_numpy(), probability, 0.5
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_index": target.index,
                    "split": split_name,
                    "y_true": target.to_numpy(),
                    "probability_failure": probability,
                    "y_pred": (probability >= chosen_threshold).astype(int),
                }
            )
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_frame.to_csv(output_dir / "threshold_analysis.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "predictions.csv", index=False
    )
    pd.DataFrame(
        {
            "feature": list(FEATURE_COLUMNS),
            "gain_importance": model.get_feature_importance(),
        }
    ).sort_values("gain_importance", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    pd.DataFrame(
        {
            "iteration": np.arange(
                1, len(model.evals_result_["training"]["binary_logloss"]) + 1
            ),
            "train_binary_logloss": model.evals_result_["training"]["binary_logloss"],
            "validation_binary_logloss": model.evals_result_["validation"]["binary_logloss"],
        }
    ).to_csv(output_dir / "learning_history.csv", index=False)

    test_roc = calculate_binary_roc_curve(
        y_test.to_numpy(), probabilities["test"], positive_label=1
    )
    pd.DataFrame(
        {
            "fpr": test_roc["fpr"],
            "tpr": test_roc["tpr"],
            "threshold": test_roc["thresholds"],
        }
    ).to_csv(output_dir / "roc_points_test.csv", index=False)
    test_pr = binary_precision_recall_curve(y_test.to_numpy(), probabilities["test"])
    pd.DataFrame(
        {
            "precision": test_pr["precision"],
            "recall": test_pr["recall"],
            "threshold": test_pr["thresholds"],
        }
    ).to_csv(output_dir / "pr_points_test.csv", index=False)

    payload = {
        "schema_version": 1,
        "scenario": "predictive_maintenance_machine_failure",
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "feature_names": list(FEATURE_COLUMNS),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "positive_rate": float(y.mean()),
        "split_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "split_class_counts": {
            name: {str(k): int(v) for k, v in target.value_counts().sort_index().items()}
            for name, (_, target) in split_data.items()
        },
        "model_params": params,
        "early_stopping_rounds": early_stopping_rounds,
        "best_iteration": int(model.best_iteration_),
        "best_validation_binary_logloss": float(model.best_score_),
        "threshold_selection_split": "validation",
        "threshold_selection_metric": "f1",
        "selected_threshold": float(chosen_threshold),
        "primary_reporting_split": "test",
        "metrics": metrics,
        "default_threshold_metrics": default_threshold_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(run_classification_case_study(), indent=2, ensure_ascii=False))
