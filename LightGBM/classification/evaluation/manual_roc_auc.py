"""Manual ROC-curve and ROC-AUC calculations for classification results.

The functions in this module operate only on labels and prediction scores.  They
do not depend on a fitted model, which keeps the evaluation layer reusable for
both native and classifier-style model APIs.
"""

from __future__ import annotations

from typing import Any, Sequence
import warnings

import numpy as np


def _as_nonempty_1d_array(values: Any, name: str) -> np.ndarray:
    """Return *values* as a non-empty one-dimensional NumPy array."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def _as_finite_score_array(values: Any, name: str) -> np.ndarray:
    """Return a one-dimensional finite floating-point score array."""

    array = _as_nonempty_1d_array(values, name)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numeric values.")
    try:
        numeric_array = array.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values.") from error
    if not np.all(np.isfinite(numeric_array)):
        raise ValueError(f"{name} must not contain NaN or infinite values.")
    return np.asarray(numeric_array, dtype=float)


def _labels_equal(left: Any, right: Any) -> bool:
    """Compare scalar labels and reject ambiguous array-like comparisons."""

    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(result, np.ndarray):
        return False
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


def _contains_nonfinite_numeric_label(values: np.ndarray) -> bool:
    """Return whether an array contains a numeric NaN or infinity label."""

    for value in values:
        if isinstance(value, (float, np.floating, complex, np.complexfloating)):
            if not bool(np.isfinite(value)):
                return True
    return False


def _validated_classes(classes: Sequence[Any]) -> list[Any]:
    """Validate class labels while preserving their supplied order."""

    class_array = _as_nonempty_1d_array(classes, "classes")
    if _contains_nonfinite_numeric_label(class_array):
        raise ValueError("classes must not contain NaN or infinite labels.")

    class_values = [
        value.item() if isinstance(value, np.generic) else value
        for value in class_array
    ]
    unique_values: list[Any] = []
    for label in class_values:
        try:
            hash(label)
        except TypeError as error:
            raise ValueError("Every class label must be hashable.") from error
        if any(_labels_equal(label, existing) for existing in unique_values):
            raise ValueError(f"classes contains the duplicate label {label!r}.")
        unique_values.append(label)
    return class_values


def _find_label_index(classes: Sequence[Any], target_label: Any) -> int:
    """Return the unique index of *target_label* in *classes*."""

    matching_indices = [
        index
        for index, class_label in enumerate(classes)
        if _labels_equal(class_label, target_label)
    ]
    if not matching_indices:
        raise ValueError(
            f"positive_label {target_label!r} is not present in classes."
        )
    if len(matching_indices) > 1:
        raise ValueError("classes must contain unique labels.")
    return matching_indices[0]


def convert_to_binary_targets(
    y_true: Sequence[Any] | np.ndarray,
    positive_label: Any,
) -> np.ndarray:
    """Convert labels to ``1`` for *positive_label* and ``0`` otherwise.

    The function intentionally does not require the positive label to occur.
    That condition is checked by :func:`calculate_binary_roc_curve`, where a
    missing positive or negative group makes the curve undefined.
    """

    labels = _as_nonempty_1d_array(y_true, "y_true")
    if _contains_nonfinite_numeric_label(labels):
        raise ValueError("y_true must not contain NaN or infinite labels.")

    return np.fromiter(
        (1 if _labels_equal(label, positive_label) else 0 for label in labels),
        dtype=np.int64,
        count=labels.size,
    )


def extract_positive_scores(
    y_proba: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    classes: Sequence[Any],
    positive_label: Any,
) -> np.ndarray:
    """Extract scores belonging to the configured positive class.

    A one-dimensional input is already interpreted as the positive-class
    probability.  For a matrix, column order must match ``classes``.
    """

    class_values = _validated_classes(classes)
    positive_index = _find_label_index(class_values, positive_label)
    probability_array = np.asarray(y_proba)

    if probability_array.ndim == 1:
        scores = _as_finite_score_array(probability_array, "y_proba")
    elif probability_array.ndim == 2:
        if probability_array.shape[0] == 0:
            raise ValueError("y_proba must not be empty.")
        if probability_array.shape[1] != len(class_values):
            raise ValueError(
                "y_proba column count must match the number of classes."
            )
        if np.iscomplexobj(probability_array):
            raise ValueError("y_proba must contain real numeric values.")
        try:
            numeric_probabilities = probability_array.astype(float, copy=False)
        except (TypeError, ValueError) as error:
            raise ValueError("y_proba must contain numeric values.") from error
        if not np.all(np.isfinite(numeric_probabilities)):
            raise ValueError("y_proba must not contain NaN or infinite values.")
        scores = np.asarray(numeric_probabilities[:, positive_index], dtype=float)
    else:
        raise ValueError("y_proba must be a one- or two-dimensional array.")

    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("y_proba values must be between 0 and 1.")
    return scores.copy()


def calculate_binary_roc_curve(
    y_true: Sequence[Any] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    positive_label: Any,
) -> dict[str, np.ndarray | float]:
    """Build a binary ROC curve by sweeping distinct scores in descending order.

    Samples sharing the same score are added as one group.  Consequently, the
    curve and its area do not depend on the original order of tied samples.

    Returns:
        A dictionary containing ``fpr``, ``tpr``, ``thresholds`` and ``auc``.

    Raises:
        ValueError: If inputs are malformed or either target group is absent.
    """

    binary_targets = convert_to_binary_targets(y_true, positive_label)
    scores = _as_finite_score_array(y_score, "y_score")
    if binary_targets.size != scores.size:
        raise ValueError("y_true and y_score must contain the same number of samples.")

    positive_count = int(binary_targets.sum())
    negative_count = int(binary_targets.size - positive_count)
    if positive_count == 0:
        raise ValueError("ROC-AUC is undefined because y_true has no positive samples.")
    if negative_count == 0:
        raise ValueError("ROC-AUC is undefined because y_true has no negative samples.")

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_targets = binary_targets[order]

    thresholds: list[float] = [float("inf")]
    true_positive_rates: list[float] = [0.0]
    false_positive_rates: list[float] = [0.0]
    cumulative_true_positives = 0
    cumulative_false_positives = 0

    group_start = 0
    while group_start < sorted_scores.size:
        threshold = float(sorted_scores[group_start])
        group_end = group_start + 1
        while (
            group_end < sorted_scores.size
            and sorted_scores[group_end] == sorted_scores[group_start]
        ):
            group_end += 1

        group_positive_count = int(sorted_targets[group_start:group_end].sum())
        group_size = group_end - group_start
        cumulative_true_positives += group_positive_count
        cumulative_false_positives += group_size - group_positive_count

        thresholds.append(threshold)
        true_positive_rates.append(cumulative_true_positives / positive_count)
        false_positive_rates.append(cumulative_false_positives / negative_count)
        group_start = group_end

    fpr = np.asarray(false_positive_rates, dtype=float)
    tpr = np.asarray(true_positive_rates, dtype=float)
    threshold_array = np.asarray(thresholds, dtype=float)
    auc = calculate_auc_trapezoid(fpr, tpr)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": threshold_array,
        "auc": auc,
    }


def calculate_auc_trapezoid(
    fpr: Sequence[float] | np.ndarray,
    tpr: Sequence[float] | np.ndarray,
) -> float:
    """Integrate a ROC curve with an explicit trapezoid loop."""

    false_positive_rates = _as_finite_score_array(fpr, "fpr")
    true_positive_rates = _as_finite_score_array(tpr, "tpr")
    if false_positive_rates.size != true_positive_rates.size:
        raise ValueError("fpr and tpr must contain the same number of points.")
    if false_positive_rates.size < 2:
        raise ValueError("At least two ROC points are required to calculate AUC.")
    if (
        np.any(false_positive_rates < 0.0)
        or np.any(false_positive_rates > 1.0)
        or np.any(true_positive_rates < 0.0)
        or np.any(true_positive_rates > 1.0)
    ):
        raise ValueError("fpr and tpr values must be between 0 and 1.")

    auc = 0.0
    for index in range(1, false_positive_rates.size):
        width = false_positive_rates[index] - false_positive_rates[index - 1]
        if width < 0.0:
            raise ValueError("fpr must be sorted in non-decreasing order.")
        left_height = true_positive_rates[index - 1]
        right_height = true_positive_rates[index]
        auc += float(width * (left_height + right_height) / 2.0)

    tolerance = 1e-12
    if auc < -tolerance or auc > 1.0 + tolerance:
        raise ValueError("Calculated AUC must be between 0 and 1.")
    return float(min(1.0, max(0.0, auc)))


def calculate_multiclass_roc_ovr(
    y_true: Sequence[Any] | np.ndarray,
    y_proba: Sequence[Sequence[float]] | np.ndarray,
    classes: Sequence[Any],
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Calculate one-vs-rest ROC curves and AUC values for every class.

    Undefined classes are retained in ``per_class`` with ``defined=False`` and
    ``auc=None``.  ``macro_auc`` is the arithmetic mean across defined classes,
    or ``None`` when no class has both positive and negative samples.
    """

    labels = _as_nonempty_1d_array(y_true, "y_true")
    if _contains_nonfinite_numeric_label(labels):
        raise ValueError("y_true must not contain NaN or infinite labels.")
    class_values = _validated_classes(classes)
    if len(class_values) < 3:
        raise ValueError("Multiclass ROC evaluation requires at least three classes.")

    probabilities = np.asarray(y_proba)
    if probabilities.ndim != 2:
        raise ValueError("y_proba must be a two-dimensional array for multiclass ROC.")
    if probabilities.shape[0] != labels.size:
        raise ValueError("y_true and y_proba must contain the same number of samples.")
    if probabilities.shape[1] != len(class_values):
        raise ValueError("y_proba column count must match the number of classes.")
    if np.iscomplexobj(probabilities):
        raise ValueError("y_proba must contain real numeric values.")
    try:
        probabilities = probabilities.astype(float, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("y_proba must contain numeric values.") from error
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("y_proba must not contain NaN or infinite values.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("y_proba values must be between 0 and 1.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("Every y_proba row must sum approximately to 1.")

    for label in labels:
        if not any(_labels_equal(label, class_label) for class_label in class_values):
            raise ValueError(f"y_true contains label {label!r} outside classes.")

    if class_names is None:
        resolved_names = [str(label) for label in class_values]
    else:
        if isinstance(class_names, (str, bytes)):
            raise ValueError("class_names must be a sequence with one name per class.")
        resolved_names = list(class_names)
        if len(resolved_names) != len(class_values):
            raise ValueError("class_names length must match the number of classes.")
        if any(not isinstance(name, str) or not name.strip() for name in resolved_names):
            raise ValueError("Every class name must be a non-empty string.")

    per_class: dict[Any, dict[str, Any]] = {}
    undefined_classes: list[Any] = []
    defined_auc_values: list[float] = []

    for class_index, class_label in enumerate(class_values):
        binary_targets = convert_to_binary_targets(labels, class_label)
        positive_count = int(binary_targets.sum())
        negative_count = int(binary_targets.size - positive_count)

        if positive_count == 0 or negative_count == 0:
            missing_group = "positive" if positive_count == 0 else "negative"
            reason = (
                f"ROC-AUC for class {class_label!r} is undefined because "
                f"there are no {missing_group} samples."
            )
            warnings.warn(reason, RuntimeWarning, stacklevel=2)
            undefined_classes.append(class_label)
            per_class[class_label] = {
                "class_label": class_label,
                "class_name": resolved_names[class_index],
                "defined": False,
                "reason": reason,
                "fpr": np.asarray([], dtype=float),
                "tpr": np.asarray([], dtype=float),
                "thresholds": np.asarray([], dtype=float),
                "auc": None,
            }
            continue

        roc_result = calculate_binary_roc_curve(
            labels,
            probabilities[:, class_index],
            class_label,
        )
        class_auc = float(roc_result["auc"])
        defined_auc_values.append(class_auc)
        per_class[class_label] = {
            "class_label": class_label,
            "class_name": resolved_names[class_index],
            "defined": True,
            "reason": None,
            "fpr": roc_result["fpr"],
            "tpr": roc_result["tpr"],
            "thresholds": roc_result["thresholds"],
            "auc": class_auc,
        }

    macro_auc = (
        float(sum(defined_auc_values) / len(defined_auc_values))
        if defined_auc_values
        else None
    )
    return {
        "per_class": per_class,
        "macro_auc": macro_auc,
        "undefined_classes": undefined_classes,
    }


__all__ = [
    "convert_to_binary_targets",
    "extract_positive_scores",
    "calculate_binary_roc_curve",
    "calculate_auc_trapezoid",
    "calculate_multiclass_roc_ovr",
]
