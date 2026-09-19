"""Generate report-ready Phase 2 figures from exported experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classification.case_study_pipeline import binary_metrics


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "case_study" / "artifacts"


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def regression_figures():
    base = ARTIFACTS / "regression"
    figures = base / "figures"
    pred = pd.read_csv(base / "predictions.csv")
    imp = pd.read_csv(base / "feature_importance.csv")
    history = pd.read_csv(base / "learning_history.csv")
    test = pred[pred["split"] == "test"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(test["y_true"], test["y_pred"], s=8, alpha=0.25)
    low = min(test["y_true"].min(), test["y_pred"].min())
    high = max(test["y_true"].max(), test["y_pred"].max())
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual exam score")
    ax.set_ylabel("Predicted exam score")
    ax.set_title("Regression: Actual vs Predicted (Test)")
    _save(fig, figures / "actual_vs_predicted_test.png")

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(test["y_pred"], test["residual"], s=8, alpha=0.25)
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Predicted exam score")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Regression: Residual Plot (Test)")
    _save(fig, figures / "residual_plot_test.png")

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.hist(test["residual"], bins=40)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Regression: Residual Distribution (Test)")
    _save(fig, figures / "residual_distribution_test.png")

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    top = imp.head(12).sort_values("gain_importance")
    ax.barh(top["feature"], top["gain_importance"])
    ax.set_xlabel("Gain importance")
    ax.set_title("Regression: Top Feature Importance")
    _save(fig, figures / "feature_importance_top12.png")

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(history["iteration"], history["train_rmse"], label="Train")
    ax.plot(history["iteration"], history["validation_rmse"], label="Validation")
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("RMSE")
    ax.set_title("Regression Learning Curve")
    ax.legend()
    _save(fig, figures / "learning_curve_rmse.png")


def classification_figures_and_default_threshold_metrics():
    base = ARTIFACTS / "classification"
    figures = base / "figures"
    pred = pd.read_csv(base / "predictions.csv")
    imp = pd.read_csv(base / "feature_importance.csv")
    history = pd.read_csv(base / "learning_history.csv")
    thresholds = pd.read_csv(base / "threshold_analysis.csv")
    roc = pd.read_csv(base / "roc_points_test.csv")
    pr = pd.read_csv(base / "pr_points_test.csv")
    metrics_path = base / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    default_metrics = {}
    for split in ("validation", "test"):
        subset = pred[pred["split"] == split]
        default_metrics[split] = binary_metrics(
            subset["y_true"].to_numpy(),
            subset["probability_failure"].to_numpy(),
            0.5,
        )
    payload["default_threshold_metrics"] = default_metrics
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    class_counts = payload["class_counts"]
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.bar(["No failure", "Failure"], [class_counts["0"], class_counts["1"]])
    ax.set_ylabel("Samples")
    ax.set_title("Classification: Class Distribution")
    _save(fig, figures / "class_distribution.png")

    test_metrics = payload["metrics"]["test"]
    matrix = np.array(
        [[test_metrics["tn"], test_metrics["fp"]],
         [test_metrics["fn"], test_metrics["tp"]]],
        dtype=int,
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title("Classification: Confusion Matrix (Test)")
    _save(fig, figures / "confusion_matrix_test.png")

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(roc["fpr"], roc["tpr"], label=f"AUC={test_metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Classification: ROC Curve (Test)")
    ax.legend()
    _save(fig, figures / "roc_curve_test.png")

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(pr["recall"], pr["precision"], label=f"AP={test_metrics['pr_auc_average_precision']:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Classification: Precision-Recall Curve (Test)")
    ax.legend()
    _save(fig, figures / "precision_recall_curve_test.png")

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(thresholds["threshold"], thresholds["precision"], label="Precision")
    ax.plot(thresholds["threshold"], thresholds["recall"], label="Recall")
    ax.plot(thresholds["threshold"], thresholds["f1"], label="F1")
    ax.axvline(payload["selected_threshold"], linestyle="--", label="Selected")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Validation Threshold Analysis")
    ax.legend()
    _save(fig, figures / "threshold_analysis_validation.png")

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(history["iteration"], history["train_binary_logloss"], label="Train")
    ax.plot(history["iteration"], history["validation_binary_logloss"], label="Validation")
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("Binary log-loss")
    ax.set_title("Classification Learning Curve")
    ax.legend()
    _save(fig, figures / "learning_curve_logloss.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    top = imp.sort_values("gain_importance")
    ax.barh(top["feature"], top["gain_importance"])
    ax.set_xlabel("Gain importance")
    ax.set_title("Classification: Feature Importance")
    _save(fig, figures / "feature_importance.png")


if __name__ == "__main__":
    regression_figures()
    classification_figures_and_default_threshold_metrics()
