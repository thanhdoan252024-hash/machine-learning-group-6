"""Manual classification metrics for binary and multiclass tasks.

The functions in this module operate only on labels and confusion-matrix
counts.  Rows of a confusion matrix represent true labels and columns
represent predicted labels.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, overload

import numpy as np

from .input_validation import (
    validate_binary_configuration,
    validate_class_names,
    validate_classes,
    validate_label_arrays,
)


@overload
def safe_divide(
    numerator: float,
    denominator: float,
    undefined_value: float = 0.0,
    *,
    return_status: bool = False,
) -> float: ...


@overload
def safe_divide(
    numerator: float,
    denominator: float,
    undefined_value: float = 0.0,
    *,
    return_status: bool,
) -> float | tuple[float, bool]: ...


def safe_divide(
    numerator: float,
    denominator: float,
    undefined_value: float = 0.0,
    *,
    return_status: bool = False,
) -> float | tuple[float, bool]:
    """Divide two finite scalars without raising on a zero denominator.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        undefined_value: Value used when ``denominator`` is zero.
        return_status: Also return whether the result was undefined when true.

    Returns:
        A float by default.  If ``return_status`` is true, returns
        ``(value, undefined)``.

    Raises:
        ValueError: If the numerator or denominator is not a finite scalar.
    """

    numerator_value = _as_finite_float(numerator, "numerator")
    denominator_value = _as_finite_float(denominator, "denominator")
    fallback_value = _as_finite_float(undefined_value, "undefined_value")

    undefined = denominator_value == 0.0
    value = fallback_value if undefined else numerator_value / denominator_value
    if return_status:
        return value, undefined
    return value


def build_confusion_matrix(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    classes: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """Build a count confusion matrix manually.

    The class order supplied by ``classes`` is preserved.  Each row is a true
    class and each column is a predicted class.
    """

    true_array, predicted_array = validate_label_arrays(y_true, y_pred)
    class_array = validate_classes(classes, true_array, predicted_array)
    class_to_index = {
        _hashable_label(label): index for index, label in enumerate(class_array)
    }
    matrix = np.zeros((len(class_array), len(class_array)), dtype=np.int64)

    for true_label, predicted_label in zip(true_array, predicted_array):
        true_index = class_to_index[_hashable_label(true_label)]
        predicted_index = class_to_index[_hashable_label(predicted_label)]
        matrix[true_index, predicted_index] += 1

    return matrix


def normalize_confusion_matrix(
    confusion_matrix: Sequence[Sequence[float]] | np.ndarray,
    mode: str = "true",
) -> np.ndarray:
    """Normalize confusion-matrix rows by true-class support.

    A row containing no samples remains all zeros.  Only ``mode='true'`` is
    supported so that the normalization convention is always explicit.
    """

    matrix = _validate_confusion_matrix(confusion_matrix)
    if mode != "true":
        raise ValueError("mode must be 'true'.")

    normalized = np.zeros(matrix.shape, dtype=float)
    for row_index in range(matrix.shape[0]):
        row_total = int(matrix[row_index, :].sum())
        if row_total != 0:
            for column_index in range(matrix.shape[1]):
                normalized[row_index, column_index] = (
                    float(matrix[row_index, column_index]) / row_total
                )
    return normalized


def calculate_ovr_counts(
    confusion_matrix: Sequence[Sequence[float]] | np.ndarray,
    class_index: int,
) -> dict[str, int]:
    """Calculate TP, TN, FP and FN for one class using one-vs-rest."""

    matrix = _validate_confusion_matrix(confusion_matrix)
    if isinstance(class_index, (bool, np.bool_)) or not isinstance(
        class_index, (int, np.integer)
    ):
        raise ValueError("class_index must be an integer.")
    index = int(class_index)
    if index < 0 or index >= matrix.shape[0]:
        raise ValueError(
            f"class_index {index} is outside the valid range "
            f"[0, {matrix.shape[0] - 1}]."
        )

    true_positive = int(matrix[index, index])
    false_negative = int(matrix[index, :].sum()) - true_positive
    false_positive = int(matrix[:, index].sum()) - true_positive
    total = int(matrix.sum())
    true_negative = total - true_positive - false_negative - false_positive

    return {
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
    }


def calculate_accuracy_from_confusion_matrix(
    confusion_matrix: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """Calculate accuracy as correct predictions divided by all samples."""

    matrix = _validate_confusion_matrix(confusion_matrix)
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("Cannot calculate accuracy from an empty confusion matrix.")

    correct = 0
    for index in range(matrix.shape[0]):
        correct += int(matrix[index, index])
    return float(correct / total)


def calculate_precision(tp: float, fp: float) -> float:
    """Calculate precision from true-positive and false-positive counts."""

    true_positive = _as_nonnegative_float(tp, "tp")
    false_positive = _as_nonnegative_float(fp, "fp")
    return float(safe_divide(true_positive, true_positive + false_positive))


def calculate_recall(tp: float, fn: float) -> float:
    """Calculate recall from true-positive and false-negative counts."""

    true_positive = _as_nonnegative_float(tp, "tp")
    false_negative = _as_nonnegative_float(fn, "fn")
    return float(safe_divide(true_positive, true_positive + false_negative))


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate the harmonic mean of precision and recall."""

    precision_value = _as_rate(precision, "precision")
    recall_value = _as_rate(recall, "recall")
    return float(
        safe_divide(
            2.0 * precision_value * recall_value,
            precision_value + recall_value,
        )
    )


def calculate_per_class_metrics(
    confusion_matrix: Sequence[Sequence[float]] | np.ndarray,
    classes: Sequence[Any] | np.ndarray,
    class_names: Sequence[str] | np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Calculate one-vs-rest metrics and support for every class.

    Undefined precision, recall, or F1 values are represented by ``0.0`` and
    accompanied by a corresponding boolean ``*_undefined`` field.
    """

    matrix = _validate_confusion_matrix(confusion_matrix)
    class_array = validate_classes(classes)
    if matrix.shape[0] != len(class_array):
        raise ValueError(
            "Confusion-matrix size must match the number of classes: "
            f"got {matrix.shape[0]} and {len(class_array)}."
        )
    name_array = validate_class_names(class_names, class_array)

    metrics: list[dict[str, Any]] = []
    for class_index, (class_label, class_name) in enumerate(
        zip(class_array, name_array)
    ):
        counts = calculate_ovr_counts(matrix, class_index)
        precision, precision_undefined = safe_divide(
            counts["tp"],
            counts["tp"] + counts["fp"],
            return_status=True,
        )
        recall, recall_undefined = safe_divide(
            counts["tp"],
            counts["tp"] + counts["fn"],
            return_status=True,
        )
        f1_score, f1_undefined = safe_divide(
            2.0 * precision * recall,
            precision + recall,
            return_status=True,
        )

        metrics.append(
            {
                "class_label": _to_python_scalar(class_label),
                "class_name": str(class_name),
                **counts,
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "support": counts["tp"] + counts["fn"],
                "precision_undefined": bool(precision_undefined),
                "recall_undefined": bool(recall_undefined),
                "f1_undefined": bool(f1_undefined),
            }
        )

    return metrics


def calculate_aggregate_metrics(
    per_class_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Calculate macro and support-weighted classification metrics."""

    if isinstance(per_class_metrics, (str, bytes)) or not isinstance(
        per_class_metrics, Sequence
    ):
        raise ValueError("per_class_metrics must be a non-empty sequence.")
    if len(per_class_metrics) == 0:
        raise ValueError("per_class_metrics must not be empty.")

    metric_names = ("precision", "recall", "f1_score")
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    supports: list[float] = []

    for item_index, item in enumerate(per_class_metrics):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"per_class_metrics[{item_index}] must be a mapping."
            )
        for metric_name in metric_names:
            if metric_name not in item:
                raise ValueError(
                    f"per_class_metrics[{item_index}] is missing "
                    f"'{metric_name}'."
                )
            values[metric_name].append(
                _as_rate(item[metric_name], f"{metric_name} at index {item_index}")
            )
        if "support" not in item:
            raise ValueError(
                f"per_class_metrics[{item_index}] is missing 'support'."
            )
        supports.append(
            _as_nonnegative_float(item["support"], f"support at index {item_index}")
        )

    total_support = sum(supports)
    if total_support == 0.0:
        raise ValueError("Total class support must be greater than zero.")

    result: dict[str, float] = {}
    number_of_classes = len(per_class_metrics)
    output_names = {
        "precision": "precision",
        "recall": "recall",
        "f1_score": "f1",
    }
    for metric_name in metric_names:
        output_name = output_names[metric_name]
        result[f"{output_name}_macro"] = float(
            sum(values[metric_name]) / number_of_classes
        )
        weighted_sum = 0.0
        for metric_value, support in zip(values[metric_name], supports):
            weighted_sum += metric_value * support
        result[f"{output_name}_weighted"] = float(weighted_sum / total_support)

    return result


def evaluate_classification(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    classes: Sequence[Any] | np.ndarray,
    class_names: Sequence[str] | np.ndarray | None = None,
    positive_label: Any | None = None,
) -> dict[str, Any]:
    """Run all label-based manual classification evaluations.

    Exactly two classes are interpreted as a binary task and require an
    explicit ``positive_label``.  Three or more classes are interpreted as a
    multiclass task.  Probability-based ROC metrics are intentionally handled
    by a separate module.
    """

    true_array, predicted_array = validate_label_arrays(y_true, y_pred)
    class_array = validate_classes(classes, true_array, predicted_array)
    if len(class_array) < 2:
        raise ValueError("Classification evaluation requires at least two classes.")
    name_array = validate_class_names(class_names, class_array)

    task_type = "binary" if len(class_array) == 2 else "multiclass"
    positive_index: int | None = None
    if task_type == "binary":
        positive_index = validate_binary_configuration(class_array, positive_label)
    elif positive_label is not None:
        raise ValueError("positive_label is only valid for binary classification.")

    matrix = build_confusion_matrix(true_array, predicted_array, class_array)
    normalized_matrix = normalize_confusion_matrix(matrix)
    accuracy = calculate_accuracy_from_confusion_matrix(matrix)
    per_class = calculate_per_class_metrics(matrix, class_array, name_array)
    aggregates = calculate_aggregate_metrics(per_class)

    result: dict[str, Any] = {
        "task_type": task_type,
        "accuracy": accuracy,
        **aggregates,
        "confusion_matrix": matrix,
        "normalized_confusion_matrix": normalized_matrix,
        "per_class_metrics": per_class,
    }

    if task_type == "binary":
        assert positive_index is not None
        positive_metrics = per_class[positive_index]
        result.update(
            {
                "positive_label": _to_python_scalar(class_array[positive_index]),
                "precision": positive_metrics["precision"],
                "recall": positive_metrics["recall"],
                "f1_score": positive_metrics["f1_score"],
                "precision_undefined": positive_metrics["precision_undefined"],
                "recall_undefined": positive_metrics["recall_undefined"],
                "f1_undefined": positive_metrics["f1_undefined"],
            }
        )

    return result


def _validate_confusion_matrix(
    confusion_matrix: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return a validated integer count confusion matrix."""

    try:
        matrix = np.asarray(confusion_matrix)
    except (TypeError, ValueError) as error:
        raise ValueError("confusion_matrix must be a rectangular numeric array.") from error
    if matrix.ndim != 2:
        raise ValueError("confusion_matrix must be two-dimensional.")
    if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion_matrix must be a non-empty square matrix.")

    try:
        numeric_matrix = matrix.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("confusion_matrix must contain numeric counts.") from error
    if not np.all(np.isfinite(numeric_matrix)):
        raise ValueError("confusion_matrix must not contain NaN or infinity.")
    if np.any(numeric_matrix < 0.0):
        raise ValueError("confusion_matrix counts must be non-negative.")
    if not np.all(numeric_matrix == np.floor(numeric_matrix)):
        raise ValueError("confusion_matrix must contain integer counts.")
    return numeric_matrix.astype(np.int64)


def _as_finite_float(value: Any, name: str) -> float:
    """Convert a scalar to a finite float with a clear validation error."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite numeric scalar.")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite numeric scalar.") from error
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite.")
    return converted


def _as_nonnegative_float(value: Any, name: str) -> float:
    """Validate and return a non-negative finite scalar."""

    converted = _as_finite_float(value, name)
    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return converted


def _as_rate(value: Any, name: str) -> float:
    """Validate and return a number in the closed interval zero to one."""

    converted = _as_finite_float(value, name)
    if converted < 0.0 or converted > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return converted


def _hashable_label(label: Any) -> Any:
    """Convert NumPy scalars and reject labels unsuitable as mapping keys."""

    converted = _to_python_scalar(label)
    try:
        hash(converted)
    except TypeError as error:
        raise ValueError("Class labels must be hashable scalar values.") from error
    return converted


def _to_python_scalar(value: Any) -> Any:
    """Convert a NumPy scalar to its equivalent built-in Python value."""

    return value.item() if isinstance(value, np.generic) else value


__all__ = [
    "safe_divide",
    "build_confusion_matrix",
    "normalize_confusion_matrix",
    "calculate_ovr_counts",
    "calculate_accuracy_from_confusion_matrix",
    "calculate_precision",
    "calculate_recall",
    "calculate_f1_score",
    "calculate_per_class_metrics",
    "calculate_aggregate_metrics",
    "evaluate_classification",
]
