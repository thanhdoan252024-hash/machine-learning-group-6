"""Công cụ đánh giá classification được xây dựng không phụ thuộc sklearn."""

from .exporters import export_evaluation_results
from .input_validation import (
    validate_binary_configuration,
    validate_classes,
    validate_class_names,
    validate_evaluation_inputs,
    validate_label_arrays,
    validate_probability_array,
)
from .manual_metrics import (
    build_confusion_matrix,
    calculate_accuracy_from_confusion_matrix,
    calculate_aggregate_metrics,
    calculate_f1_score,
    calculate_ovr_counts,
    calculate_per_class_metrics,
    calculate_precision,
    calculate_recall,
    evaluate_classification,
    normalize_confusion_matrix,
    safe_divide,
)
from .manual_roc_auc import (
    calculate_auc_trapezoid,
    calculate_binary_roc_curve,
    calculate_multiclass_roc_ovr,
    convert_to_binary_targets,
    extract_positive_scores,
)
from .reports import create_classification_report_dataframe


__all__ = [
    "build_confusion_matrix",
    "calculate_accuracy_from_confusion_matrix",
    "calculate_aggregate_metrics",
    "calculate_auc_trapezoid",
    "calculate_binary_roc_curve",
    "calculate_f1_score",
    "calculate_multiclass_roc_ovr",
    "calculate_ovr_counts",
    "calculate_per_class_metrics",
    "calculate_precision",
    "calculate_recall",
    "convert_to_binary_targets",
    "create_classification_report_dataframe",
    "evaluate_classification",
    "export_evaluation_results",
    "extract_positive_scores",
    "normalize_confusion_matrix",
    "safe_divide",
    "validate_binary_configuration",
    "validate_classes",
    "validate_class_names",
    "validate_evaluation_inputs",
    "validate_label_arrays",
    "validate_probability_array",
]
