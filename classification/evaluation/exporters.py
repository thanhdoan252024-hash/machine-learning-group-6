"""Xuất kết quả đánh giá thành các tệp CSV có cấu trúc ổn định."""

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SUMMARY_METRIC_ORDER = (
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "roc_auc_macro",
)

TABLE_FLOAT_FORMAT = "%.6f"
SCORE_FLOAT_FORMAT = "%.12g"


def export_evaluation_results(
    metrics: Mapping[str, Any],
    report_df: pd.DataFrame,
    confusion_matrix: Any,
    normalized_confusion_matrix: Any,
    y_true: Any,
    y_pred: Any,
    y_proba: Any,
    classes: Sequence[Any],
    output_dir: str | Path,
    roc_results: Mapping[str, Any] | None = None,
    class_names: Sequence[str] | None = None,
    positive_label: Any | None = None,
) -> dict[str, Any]:
    """Xuất bảng metric, report, ma trận, prediction và điểm ROC.

    Args:
        metrics: Kết quả tổng hợp đã được tính bởi các module metric/ROC.
        report_df: Classification report đã được tạo sẵn.
        confusion_matrix: Confusion Matrix số lượng.
        normalized_confusion_matrix: Confusion Matrix chuẩn hóa theo nhãn thật.
        y_true: Nhãn thật một chiều.
        y_pred: Nhãn dự đoán một chiều.
        y_proba: Xác suất binary hoặc multiclass.
        classes: Nhãn lớp theo thứ tự cột xác suất.
        output_dir: Thư mục gốc chứa ``tables`` và ``predictions``.
        roc_results: Kết quả ROC binary hoặc multiclass; có thể bỏ qua.
        class_names: Tên hiển thị theo thứ tự ``classes``.
        positive_label: Bắt buộc khi xuất xác suất binary.

    Returns:
        Dictionary chứa đường dẫn của tất cả tệp đã tạo.
    """
    root, tables_dir, predictions_dir = _prepare_output_directories(output_dir)
    del root  # Tên biến giúp thể hiện cấu trúc nhưng không cần dùng tiếp.

    class_values = list(classes)
    if len(class_values) < 2:
        raise ValueError("classes phải chứa ít nhất hai lớp.")
    display_names = _normalize_class_names(class_values, class_names)

    count_matrix = _validate_matrix(
        confusion_matrix, len(class_values), "confusion_matrix"
    )
    normalized_matrix = _validate_matrix(
        normalized_confusion_matrix,
        len(class_values),
        "normalized_confusion_matrix",
    )
    true_values, predicted_values = _validate_prediction_labels(y_true, y_pred)
    probabilities = _validate_probability_shape(
        y_proba, true_values.size, len(class_values)
    )

    paths: dict[str, Any] = {}
    summary_path = tables_dir / "metrics_summary.csv"
    _create_metrics_summary(metrics).to_csv(
        summary_path, index=False, encoding="utf-8", float_format=TABLE_FLOAT_FORMAT
    )
    paths["metrics_summary"] = summary_path

    report_path = tables_dir / "classification_report.csv"
    report_df.to_csv(
        report_path,
        index=False,
        encoding="utf-8",
        float_format=TABLE_FLOAT_FORMAT,
    )
    paths["classification_report"] = report_path

    count_path = tables_dir / "confusion_matrix_counts.csv"
    _create_confusion_matrix_dataframe(count_matrix, display_names).to_csv(
        count_path, index=False, encoding="utf-8"
    )
    paths["confusion_matrix_counts"] = count_path

    normalized_path = tables_dir / "confusion_matrix_normalized.csv"
    _create_confusion_matrix_dataframe(normalized_matrix, display_names).to_csv(
        normalized_path,
        index=False,
        encoding="utf-8",
        float_format=TABLE_FLOAT_FORMAT,
    )
    paths["confusion_matrix_normalized"] = normalized_path

    predictions_path = predictions_dir / "predictions.csv"
    _create_predictions_dataframe(
        true_values,
        predicted_values,
        probabilities,
        class_values,
        display_names,
        positive_label,
    ).to_csv(
        predictions_path,
        index=False,
        encoding="utf-8",
        # Giữ đủ precision để có thể tính lại ranking/AUC từ predictions.csv.
        float_format=SCORE_FLOAT_FORMAT,
    )
    paths["predictions"] = predictions_path

    if roc_results is not None:
        paths["roc_points"] = _export_roc_points(
            roc_results, class_values, display_names, tables_dir
        )

    return paths


def _create_metrics_summary(metrics: Mapping[str, Any]) -> pd.DataFrame:
    """Chọn các metric scalar đã biết theo thứ tự ổn định."""
    rows: list[dict[str, Any]] = []
    for name in SUMMARY_METRIC_ORDER:
        if name not in metrics:
            continue
        value = metrics[name]
        if value is None:
            rows.append({"metric": name, "value": np.nan})
            continue
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise ValueError(f"Metric {name} phải là một scalar số.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Metric {name} phải là một scalar số.") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"Metric {name} phải là số hữu hạn.")
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError(f"Metric {name} phải nằm trong đoạn [0, 1].")
        rows.append({"metric": name, "value": numeric})
    if not rows:
        raise ValueError("metrics không chứa metric scalar được hỗ trợ để xuất.")
    return pd.DataFrame(rows, columns=["metric", "value"])


def _create_confusion_matrix_dataframe(
    matrix: np.ndarray, class_names: Sequence[str]
) -> pd.DataFrame:
    """Tạo DataFrame ma trận với nhãn hàng/cột có ý nghĩa."""
    column_names = _unique_prefixed_names(class_names, "predicted")
    frame = pd.DataFrame(matrix, columns=column_names)
    frame.insert(0, "true_label", list(class_names))
    return frame


def _create_predictions_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: Sequence[Any],
    class_names: Sequence[str],
    positive_label: Any | None,
) -> pd.DataFrame:
    """Tạo bảng prediction theo contract binary/multiclass."""
    data: dict[str, Any] = {
        "actual_label": y_true,
        "predicted_label": y_pred,
    }
    if len(classes) == 2:
        if positive_label is None:
            raise ValueError("positive_label là bắt buộc khi xuất binary predictions.")
        positive_index = _find_label_index(classes, positive_label)
        if positive_index is None:
            raise ValueError("positive_label phải thuộc classes.")
        positive_scores = y_proba if y_proba.ndim == 1 else y_proba[:, positive_index]
        probability_name = f"probability_{_safe_name(class_names[positive_index])}"
        data[probability_name] = positive_scores
    else:
        if y_proba.ndim != 2 or y_proba.shape[1] != len(classes):
            raise ValueError(
                "Multiclass y_proba phải có shape (n_samples, n_classes)."
            )
        probability_names = _unique_prefixed_names(class_names, "probability")
        for index, column_name in enumerate(probability_names):
            data[column_name] = y_proba[:, index]
    data["is_correct"] = y_true == y_pred
    return pd.DataFrame(data)


def _export_roc_points(
    roc_results: Mapping[str, Any],
    classes: Sequence[Any],
    class_names: Sequence[str],
    tables_dir: Path,
) -> Path | dict[Any, Path]:
    """Xuất một ROC CSV cho binary hoặc một tệp trên mỗi lớp multiclass."""
    if "per_class" not in roc_results:
        frame = _roc_frame_from_result(roc_results, "binary ROC")
        path = tables_dir / "roc_points.csv"
        frame.to_csv(
            path,
            index=False,
            encoding="utf-8",
            float_format=SCORE_FLOAT_FORMAT,
        )
        return path

    per_class = roc_results["per_class"]
    if not isinstance(per_class, Mapping):
        raise ValueError("roc_results['per_class'] phải là một mapping.")
    paths: dict[Any, Path] = {}
    for class_index, (class_label, class_name) in enumerate(zip(classes, class_names)):
        result = _lookup_class_result(per_class, class_label)
        if result is None:
            continue
        if result.get("defined", True):
            frame = _roc_frame_from_result(result, f"ROC lớp {class_label}")
        else:
            frame = pd.DataFrame(
                [
                    {
                        "threshold": np.nan,
                        "fpr": np.nan,
                        "tpr": np.nan,
                        "defined": False,
                        "reason": result.get("reason", "ROC-AUC undefined"),
                    }
                ]
            )
        path = tables_dir / f"roc_points_class_{class_index}_{_safe_name(class_name)}.csv"
        frame.to_csv(
            path,
            index=False,
            encoding="utf-8",
            float_format=SCORE_FLOAT_FORMAT,
        )
        paths[class_label] = path
    return paths


def _roc_frame_from_result(result: Mapping[str, Any], name: str) -> pd.DataFrame:
    """Tạo DataFrame threshold/FPR/TPR và kiểm tra độ dài."""
    missing = [key for key in ("thresholds", "fpr", "tpr") if key not in result]
    if missing:
        raise ValueError(f"{name} thiếu các trường: " + ", ".join(missing))
    thresholds = np.asarray(result["thresholds"], dtype=float)
    fpr = np.asarray(result["fpr"], dtype=float)
    tpr = np.asarray(result["tpr"], dtype=float)
    if thresholds.ndim != 1 or fpr.ndim != 1 or tpr.ndim != 1:
        raise ValueError(f"{name} phải chứa các mảng một chiều.")
    if not (thresholds.size == fpr.size == tpr.size):
        raise ValueError(f"thresholds, fpr và tpr của {name} phải cùng độ dài.")
    return pd.DataFrame({"threshold": thresholds, "fpr": fpr, "tpr": tpr})


def _prepare_output_directories(output_dir: str | Path) -> tuple[Path, Path, Path]:
    """Tạo cấu trúc output an toàn."""
    root = Path(output_dir)
    if root.exists() and not root.is_dir():
        raise ValueError(f"output_dir phải là thư mục: {root}")
    tables_dir = root / "tables"
    predictions_dir = root / "predictions"
    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    return root, tables_dir, predictions_dir


def _normalize_class_names(
    classes: Sequence[Any], class_names: Sequence[str] | None
) -> list[str]:
    """Chuẩn hóa tên lớp hoặc suy ra từ label."""
    names = [str(label) for label in classes] if class_names is None else [
        str(name).strip() for name in class_names
    ]
    if len(names) != len(classes) or any(not name for name in names):
        raise ValueError("class_names phải có tên không rỗng cho từng lớp.")
    return names


def _validate_matrix(matrix: Any, n_classes: int, name: str) -> np.ndarray:
    """Kiểm tra shape và giá trị ma trận trước khi xuất."""
    values = np.asarray(matrix)
    if values.shape != (n_classes, n_classes):
        raise ValueError(f"{name} phải có shape ({n_classes}, {n_classes}).")
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} phải chứa dữ liệu số.") from exc
    if not np.all(np.isfinite(numeric)) or np.any(numeric < 0):
        raise ValueError(f"{name} phải chứa số hữu hạn không âm.")
    return values


def _validate_prediction_labels(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """Kiểm tra shape nhãn phục vụ bảng predictions."""
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    if true_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("y_true và y_pred phải là mảng một chiều.")
    if true_values.size == 0 or true_values.size != predicted_values.size:
        raise ValueError("y_true và y_pred phải cùng số mẫu và không được rỗng.")
    return true_values, predicted_values


def _validate_probability_shape(
    y_proba: Any, n_samples: int, n_classes: int
) -> np.ndarray:
    """Kiểm tra shape tối thiểu của xác suất trước khi xuất."""
    probabilities = np.asarray(y_proba, dtype=float)
    valid_binary = n_classes == 2 and probabilities.shape in {
        (n_samples,),
        (n_samples, 2),
    }
    valid_multiclass = n_classes > 2 and probabilities.shape == (
        n_samples,
        n_classes,
    )
    if not valid_binary and not valid_multiclass:
        raise ValueError("y_proba không khớp số mẫu hoặc số lớp.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("y_proba không được chứa NaN hoặc Infinity.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("y_proba phải nằm trong đoạn [0, 1].")
    if probabilities.ndim == 2 and not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=1e-7, atol=1e-8
    ):
        raise ValueError("Tổng xác suất trên mỗi hàng y_proba phải xấp xỉ 1.")
    return probabilities


def _find_label_index(classes: Sequence[Any], target: Any) -> int | None:
    """Tìm label bằng phép so sánh an toàn cho NumPy scalar."""
    for index, label in enumerate(classes):
        try:
            if bool(label == target):
                return index
        except (TypeError, ValueError):
            continue
    return None


def _lookup_class_result(
    per_class: Mapping[Any, Any], class_label: Any
) -> Mapping[str, Any] | None:
    """Tra cứu kết quả theo label mà không giả định label là chuỗi."""
    try:
        result = per_class.get(class_label)
    except TypeError:
        result = None
    if result is not None:
        return result
    for stored_label, stored_result in per_class.items():
        try:
            if bool(stored_label == class_label):
                return stored_result
        except (TypeError, ValueError):
            continue
    return None


def _safe_name(value: Any) -> str:
    """Tạo thành phần tên cột/tệp dễ đọc mà không dùng thư viện ngoài."""
    text = str(value).strip().lower()
    cleaned = "".join(character if character.isalnum() else "_" for character in text)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "class"


def _unique_prefixed_names(values: Sequence[Any], prefix: str) -> list[str]:
    """Tạo tên cột duy nhất kể cả khi nhiều tên lớp chuẩn hóa giống nhau."""
    used_names: set[str] = set()
    names: list[str] = []
    for value in values:
        base_name = f"{prefix}_{_safe_name(value)}"
        column_name = base_name
        suffix = 2
        while column_name in used_names:
            column_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(column_name)
        names.append(column_name)
    return names
