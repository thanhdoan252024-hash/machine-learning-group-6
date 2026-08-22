"""Trực quan hóa kết quả classification chỉ bằng Matplotlib."""

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

# Chỉ lưu hình ra tệp; backend không giao diện giúp runner hoạt động ổn định
# trên server, CI và môi trường Python không cài Tcl/Tk.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DPI = 300


def plot_confusion_matrix(
    matrix: Any,
    class_names: Sequence[str],
    normalized: bool = False,
    output_path: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Vẽ và lưu Confusion Matrix số lượng hoặc đã chuẩn hóa."""
    raw_values = np.asarray(matrix)
    names = _validate_class_names_for_square_matrix(raw_values, class_names)
    values = raw_values.astype(float)
    if not normalized:
        if not np.all(values == np.floor(values)):
            raise ValueError("Confusion Matrix số lượng phải chứa số nguyên.")
        values = values.astype(np.int64)
    path = _prepare_output_path(output_path, "confusion_matrix.png")

    figure_size = max(6.0, min(14.0, 1.15 * len(names) + 3.0))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size * 0.88))
    image = ax.imshow(values, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    positions = np.arange(len(names))
    ax.set(
        xticks=positions,
        yticks=positions,
        xticklabels=names,
        yticklabels=names,
        xlabel="Predicted label",
        ylabel="True label",
        title=(
            "Normalized Confusion Matrix"
            if normalized
            else "Confusion Matrix (Counts)"
        ),
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    maximum = float(np.max(values)) if values.size else 0.0
    threshold = maximum / 2.0
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            # Ba chữ số giúp tỷ lệ lỗi nhỏ (ví dụ 5/1932) không bị hiển thị
            # thành 0.00 và gây hiểu nhầm, trong khi CSV vẫn giữ sáu chữ số.
            text = f"{float(value):.3f}" if normalized else str(int(value))
            ax.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                color="white" if float(value) > threshold else "black",
            )

    fig.tight_layout()
    _save_and_close(fig, path, dpi)
    return path


def plot_overall_metrics_bar(
    metrics: Mapping[str, float],
    output_path: str | Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Vẽ bar chart cho các metric tổng thể đã được tính sẵn."""
    if not metrics:
        raise ValueError("metrics không được rỗng.")
    labels = [str(label) for label in metrics]
    values = np.asarray([metrics[label] for label in metrics], dtype=float)
    _validate_metric_values(values, "metrics")
    path = _prepare_output_path(output_path, "overall_metrics_bar.png")

    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 1.35), 5.5))
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Overall Classification Metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    _save_and_close(fig, path, dpi)
    return path


def plot_per_class_metrics(
    report_df: pd.DataFrame,
    output_path: str | Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Vẽ Precision, Recall và F1-score cho từng lớp trong report."""
    required_columns = {"label", "row_type", "precision", "recall", "f1_score"}
    missing = required_columns.difference(report_df.columns)
    if missing:
        raise ValueError(
            "report_df thiếu các cột: " + ", ".join(sorted(missing))
        )

    class_rows = report_df[report_df["row_type"] == "class"].copy()
    if class_rows.empty:
        raise ValueError("report_df không có dòng metric theo lớp.")

    labels = class_rows["label"].astype(str).tolist()
    precision = class_rows["precision"].to_numpy(dtype=float)
    recall = class_rows["recall"].to_numpy(dtype=float)
    f1_score = class_rows["f1_score"].to_numpy(dtype=float)
    _validate_metric_values(
        np.concatenate([precision, recall, f1_score]), "per-class metrics"
    )
    path = _prepare_output_path(output_path, "per_class_metrics.png")

    positions = np.arange(len(labels), dtype=float)
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 1.6), 5.8))
    ax.bar(positions - width, precision, width, label="Precision")
    ax.bar(positions, recall, width, label="Recall")
    ax.bar(positions + width, f1_score, width, label="F1-score")
    ax.set(
        xticks=positions,
        xticklabels=labels,
        ylim=(0.0, 1.0),
        ylabel="Score",
        title="Metrics by Class",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    _save_and_close(fig, path, dpi)
    return path


def plot_binary_roc_curve(
    fpr: Any,
    tpr: Any,
    auc: float,
    output_path: str | Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Vẽ ROC Curve binary cùng đường tham chiếu ngẫu nhiên."""
    fpr_values, tpr_values = _validate_roc_points(fpr, tpr)
    auc_value = float(auc)
    _validate_metric_values(np.asarray([auc_value]), "auc")
    path = _prepare_output_path(output_path, "roc_curve.png")

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot(
        fpr_values,
        tpr_values,
        linewidth=2,
        label=f"ROC curve (AUC = {auc_value:.3f})",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="Random")
    ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Binary ROC Curve",
    )
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_and_close(fig, path, dpi)
    return path


def plot_multiclass_roc_curves(
    roc_results: Mapping[str, Any],
    class_names: Sequence[str],
    macro_auc: float | None,
    output_path: str | Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Vẽ các ROC Curve One-vs-Rest đã được tính cho multiclass."""
    per_class = roc_results.get("per_class", roc_results)
    if not isinstance(per_class, Mapping) or not per_class:
        raise ValueError("roc_results không có kết quả ROC theo lớp.")
    if len(class_names) != len(per_class):
        raise ValueError("Số class_names phải bằng số lớp trong roc_results.")
    path = _prepare_output_path(output_path, "roc_ovr_multiclass.png")

    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    plotted = 0
    for (class_label, result), class_name in zip(per_class.items(), class_names):
        if not result.get("defined", True) or result.get("auc") is None:
            continue
        fpr_values, tpr_values = _validate_roc_points(
            result["fpr"], result["tpr"]
        )
        auc_value = float(result["auc"])
        _validate_metric_values(np.asarray([auc_value]), f"AUC lớp {class_label}")
        ax.plot(
            fpr_values,
            tpr_values,
            linewidth=2,
            label=f"{class_name} (AUC = {auc_value:.3f})",
        )
        plotted += 1

    title = "Multiclass ROC Curves (One-vs-Rest)"
    if macro_auc is not None:
        macro_value = float(macro_auc)
        _validate_metric_values(np.asarray([macro_value]), "macro_auc")
        title += f"\nMacro AUC = {macro_value:.3f}"
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="Random")
    if plotted == 0:
        ax.text(
            0.5,
            0.42,
            "ROC-AUC undefined\n(no class has both positive and negative samples)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=title,
    )
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_and_close(fig, path, dpi)
    return path


def plot_correct_incorrect_pie(
    y_true: Any,
    y_pred: Any,
    output_path: str | Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Vẽ tỷ lệ dự đoán đúng/sai; không thay thế các metric phân loại."""
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    if true_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("y_true và y_pred phải là mảng một chiều.")
    if true_values.size == 0:
        raise ValueError("y_true và y_pred không được rỗng.")
    if true_values.size != predicted_values.size:
        raise ValueError("y_true và y_pred phải có cùng số mẫu.")

    correct = int(np.sum(true_values == predicted_values))
    incorrect = int(true_values.size - correct)
    path = _prepare_output_path(output_path, "correct_incorrect_pie.png")

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.pie(
        [correct, incorrect],
        labels=["Correct", "Incorrect"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#59A14F", "#E15759"],
        wedgeprops={"edgecolor": "white"},
    )
    ax.set_title("Correct Predictions vs Incorrect Predictions")
    ax.axis("equal")
    fig.tight_layout()
    _save_and_close(fig, path, dpi)
    return path


def _validate_class_names_for_square_matrix(
    matrix: np.ndarray, class_names: Sequence[str]
) -> list[str]:
    """Kiểm tra shape Confusion Matrix và tên lớp."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Confusion Matrix phải là ma trận vuông hai chiều.")
    if matrix.shape[0] == 0:
        raise ValueError("Confusion Matrix không được rỗng.")
    try:
        numeric_matrix = matrix.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Confusion Matrix phải chứa dữ liệu số.") from exc
    if not np.all(np.isfinite(numeric_matrix)):
        raise ValueError("Confusion Matrix không được chứa NaN hoặc Infinity.")
    if np.any(numeric_matrix < 0):
        raise ValueError("Confusion Matrix không được chứa giá trị âm.")
    names = [str(name).strip() for name in class_names]
    if len(names) != matrix.shape[0] or any(not name for name in names):
        raise ValueError("class_names phải có tên không rỗng cho từng lớp.")
    return names


def _validate_metric_values(values: np.ndarray, name: str) -> None:
    """Kiểm tra dãy metric hữu hạn trong đoạn [0, 1]."""
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} phải chứa các số hữu hạn.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} phải nằm trong đoạn [0, 1].")


def _validate_roc_points(fpr: Any, tpr: Any) -> tuple[np.ndarray, np.ndarray]:
    """Kiểm tra hai mảng điểm ROC trước khi vẽ."""
    fpr_values = np.asarray(fpr, dtype=float)
    tpr_values = np.asarray(tpr, dtype=float)
    if fpr_values.ndim != 1 or tpr_values.ndim != 1:
        raise ValueError("fpr và tpr phải là mảng một chiều.")
    if fpr_values.size < 2 or fpr_values.size != tpr_values.size:
        raise ValueError("fpr và tpr phải cùng độ dài và có ít nhất hai điểm.")
    _validate_metric_values(fpr_values, "fpr")
    _validate_metric_values(tpr_values, "tpr")
    return fpr_values, tpr_values


def _prepare_output_path(
    output_path: str | Path | None, default_name: str
) -> Path:
    """Chuẩn hóa đường dẫn và tạo thư mục cha khi cần."""
    path = Path(default_name if output_path is None else output_path)
    if path.suffix.lower() != ".png":
        raise ValueError(f"Đường dẫn hình phải có đuôi .png: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_and_close(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    """Lưu figure an toàn và luôn đóng tài nguyên Matplotlib."""
    if isinstance(dpi, bool) or not isinstance(dpi, (int, np.integer)) or dpi <= 0:
        plt.close(fig)
        raise ValueError("dpi phải là số nguyên dương.")
    try:
        fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    finally:
        plt.close(fig)
