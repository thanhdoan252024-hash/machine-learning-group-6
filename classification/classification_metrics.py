"""Facade cho bộ metric classification được cài đặt thủ công.

File này được giữ ở vị trí cũ để notebook hoặc thành viên trong nhóm có thể
import API quen thuộc. Công thức thật nằm trong package
:mod:`classification.evaluation` và
không sử dụng ``sklearn.metrics`` hay metric có sẵn của LightGBM.
"""

from classification.evaluation import (
    build_confusion_matrix,
    calculate_accuracy_from_confusion_matrix,
    calculate_aggregate_metrics,
    calculate_auc_trapezoid,
    calculate_binary_roc_curve,
    calculate_f1_score,
    calculate_ovr_counts,
    calculate_per_class_metrics,
    calculate_precision,
    calculate_recall,
    create_classification_report_dataframe,
    evaluate_classification,
    normalize_confusion_matrix,
)


__all__ = [
    "build_confusion_matrix",
    "calculate_accuracy_from_confusion_matrix",
    "calculate_aggregate_metrics",
    "calculate_auc_trapezoid",
    "calculate_binary_roc_curve",
    "calculate_f1_score",
    "calculate_ovr_counts",
    "calculate_per_class_metrics",
    "calculate_precision",
    "calculate_recall",
    "create_classification_report_dataframe",
    "evaluate_classification",
    "normalize_confusion_matrix",
]
