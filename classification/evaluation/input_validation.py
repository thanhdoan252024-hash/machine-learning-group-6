"""Validation helpers for classification evaluation inputs.

The functions in this module validate the hand-off contract between a trained
classifier and the evaluation pipeline.  They only validate and expose NumPy
views of the supplied data: invalid probabilities are never clipped,
renormalized, reshaped, or otherwise repaired.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


_VALID_TASK_TYPES = {"binary", "multiclass"}


def _as_one_dimensional_object_array(values: Sequence[Any], name: str) -> np.ndarray:
    """Return *values* as an object array while preserving scalar label types."""

    try:
        array = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a one-dimensional array.") from exc

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; received an array with "
            f"{array.ndim} dimensions and shape {array.shape}."
        )
    return array


def _validate_label_values(values: np.ndarray, name: str) -> None:
    """Validate that an array contains finite numeric or string scalar labels."""

    for index, value in enumerate(values):
        if isinstance(value, (str, np.str_)):
            continue

        if isinstance(value, (int, np.integer, bool, np.bool_)):
            continue

        if isinstance(value, (float, np.floating)):
            if bool(np.isnan(value)):
                raise ValueError(f"{name} contains NaN at index {index}.")
            if bool(np.isinf(value)):
                raise ValueError(f"{name} contains infinity at index {index}.")
            continue

        raise ValueError(
            f"{name} must contain only finite numeric or string labels; "
            f"found {type(value).__name__} at index {index}."
        )


def _labels_equal(left: Any, right: Any) -> bool:
    """Compare two already-validated scalar labels without NumPy ambiguity."""

    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _contains_label(classes: np.ndarray, label: Any) -> bool:
    """Return whether *classes* contains *label* using scalar equality."""

    return any(_labels_equal(class_label, label) for class_label in classes)


def validate_label_arrays(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return the true and predicted label vectors.

    Both inputs must be non-empty, one-dimensional vectors with the same
    length.  Supported labels are finite real numbers and strings.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.

    Returns:
        A tuple ``(validated_y_true, validated_y_pred)``.  Object dtype is used
        so a supplied label's Python type is not silently coerced by NumPy.

    Raises:
        ValueError: If a vector has an invalid shape, length, or label value.
    """

    true_array = _as_one_dimensional_object_array(y_true, "y_true")
    predicted_array = _as_one_dimensional_object_array(y_pred, "y_pred")

    if true_array.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if predicted_array.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if true_array.size != predicted_array.size:
        raise ValueError(
            "y_true and y_pred must contain the same number of samples; "
            f"received {true_array.size} and {predicted_array.size}."
        )

    _validate_label_values(true_array, "y_true")
    _validate_label_values(predicted_array, "y_pred")
    return true_array, predicted_array


def validate_classes(
    classes: Sequence[Any],
    y_true: Sequence[Any] | None = None,
    y_pred: Sequence[Any] | None = None,
) -> np.ndarray:
    """Validate the ordered class list and optional observed labels.

    The order supplied in ``classes`` is retained because it defines both
    confusion-matrix axes and probability-matrix columns.

    Args:
        classes: Ordered, unique class labels.
        y_true: Optional true labels that must all occur in ``classes``.
        y_pred: Optional predicted labels that must all occur in ``classes``.

    Returns:
        The validated one-dimensional class array, in the supplied order.

    Raises:
        ValueError: If classes are empty, duplicated, invalid, or do not cover
            every supplied true and predicted label.
    """

    class_array = _as_one_dimensional_object_array(classes, "classes")
    if class_array.size == 0:
        raise ValueError("classes must not be empty.")
    _validate_label_values(class_array, "classes")

    for index, label in enumerate(class_array):
        for previous_index in range(index):
            if _labels_equal(label, class_array[previous_index]):
                raise ValueError(
                    "classes must contain unique labels; "
                    f"duplicate label {label!r} found at indices "
                    f"{previous_index} and {index}."
                )

    for values, name in ((y_true, "y_true"), (y_pred, "y_pred")):
        if values is None:
            continue
        observed = _as_one_dimensional_object_array(values, name)
        _validate_label_values(observed, name)
        for index, label in enumerate(observed):
            if not _contains_label(class_array, label):
                raise ValueError(
                    f"{name} contains label {label!r} at index {index}, "
                    "which is not present in classes."
                )

    return class_array


def validate_class_names(
    class_names: Sequence[str] | None,
    classes: Sequence[Any] | int,
) -> np.ndarray:
    """Validate display names corresponding positionally to ``classes``.

    Passing ``None`` creates display names with ``str(class_label)``.  Explicit
    names must be strings, and blank or whitespace-only names are rejected.

    Args:
        class_names: Ordered display names, or ``None`` to derive them.
        classes: The class-label sequence, or its positive integer length.

    Returns:
        A one-dimensional object array of display names.

    Raises:
        ValueError: If the names have an invalid shape, count, or value.
    """

    if isinstance(classes, (int, np.integer)) and not isinstance(
        classes, (bool, np.bool_)
    ):
        n_classes = int(classes)
        if n_classes <= 0:
            raise ValueError("The number of classes must be greater than zero.")
        class_array: np.ndarray | None = None
    else:
        class_array = validate_classes(classes)  # type: ignore[arg-type]
        n_classes = int(class_array.size)

    if class_names is None:
        if class_array is None:
            raise ValueError(
                "class_names cannot be None when only the number of classes is supplied."
            )
        return np.asarray([str(label) for label in class_array], dtype=object)

    name_array = _as_one_dimensional_object_array(class_names, "class_names")
    if name_array.size != n_classes:
        raise ValueError(
            "class_names must contain exactly one name for each class; "
            f"received {name_array.size} names for {n_classes} classes."
        )

    for index, name in enumerate(name_array):
        if not isinstance(name, (str, np.str_)):
            raise ValueError(
                "class_names must contain only strings; "
                f"found {type(name).__name__} at index {index}."
            )
        if not str(name).strip():
            raise ValueError(f"class_names contains an empty name at index {index}.")

    return name_array


def validate_probability_array(
    y_proba: Sequence[Any] | np.ndarray,
    n_samples: int,
    n_classes: int,
    task_type: str,
) -> np.ndarray:
    """Validate binary or multiclass prediction probabilities.

    Binary probabilities may be a vector containing the declared positive
    class probability or a two-column matrix.  Multiclass probabilities must
    be a matrix with one column per class.  Every matrix row must sum to one
    within NumPy's standard ``allclose`` tolerance.

    Args:
        y_proba: Probability vector or matrix.
        n_samples: Expected number of rows/samples.
        n_classes: Number of declared classes.
        task_type: Either ``"binary"`` or ``"multiclass"``.

    Returns:
        The validated probability array without clipping or normalization.

    Raises:
        ValueError: If dtype, shape, finiteness, range, or row sums are invalid.
    """

    if not isinstance(n_samples, (int, np.integer)) or isinstance(
        n_samples, (bool, np.bool_)
    ):
        raise ValueError("n_samples must be a positive integer.")
    if int(n_samples) <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if not isinstance(n_classes, (int, np.integer)) or isinstance(
        n_classes, (bool, np.bool_)
    ):
        raise ValueError("n_classes must be an integer greater than one.")
    if int(n_classes) < 2:
        raise ValueError("n_classes must be an integer greater than one.")
    if task_type not in _VALID_TASK_TYPES:
        raise ValueError(
            "task_type must be either 'binary' or 'multiclass'; "
            f"received {task_type!r}."
        )

    try:
        probability_array = np.asarray(y_proba)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_proba must be convertible to a numeric NumPy array.") from exc

    if probability_array.ndim not in (1, 2):
        raise ValueError(
            "y_proba must be one- or two-dimensional; "
            f"received {probability_array.ndim} dimensions and shape "
            f"{probability_array.shape}."
        )
    if probability_array.shape[0] != int(n_samples):
        raise ValueError(
            "y_proba must contain one row/value per sample; "
            f"received {probability_array.shape[0]} for {n_samples} samples."
        )
    if (
        not np.issubdtype(probability_array.dtype, np.number)
        or np.issubdtype(probability_array.dtype, np.bool_)
        or np.issubdtype(probability_array.dtype, np.complexfloating)
    ):
        raise ValueError(
            "y_proba must contain real numeric probabilities; "
            f"received dtype {probability_array.dtype}."
        )

    finite_mask = np.isfinite(probability_array)
    if not bool(np.all(finite_mask)):
        first_index = tuple(int(part) for part in np.argwhere(~finite_mask)[0])
        invalid_value = probability_array[first_index]
        description = "NaN" if bool(np.isnan(invalid_value)) else "infinity"
        raise ValueError(f"y_proba contains {description} at index {first_index}.")

    in_range_mask = (probability_array >= 0.0) & (probability_array <= 1.0)
    if not bool(np.all(in_range_mask)):
        first_index = tuple(int(part) for part in np.argwhere(~in_range_mask)[0])
        raise ValueError(
            "y_proba values must lie in the inclusive interval [0, 1]; "
            f"found {probability_array[first_index]!r} at index {first_index}."
        )

    if task_type == "binary":
        if int(n_classes) != 2:
            raise ValueError(
                "Binary probability validation requires exactly two classes; "
                f"received {n_classes}."
            )
        if probability_array.ndim == 2 and probability_array.shape[1] != 2:
            raise ValueError(
                "Binary y_proba matrices must have exactly 2 columns; "
                f"received shape {probability_array.shape}."
            )
    else:
        if int(n_classes) < 3:
            raise ValueError(
                "Multiclass probability validation requires at least three classes; "
                f"received {n_classes}."
            )
        if probability_array.ndim != 2:
            raise ValueError(
                "Multiclass y_proba must be a two-dimensional matrix with one "
                "column per class."
            )
        if probability_array.shape[1] != int(n_classes):
            raise ValueError(
                "Multiclass y_proba column count must match n_classes; "
                f"received {probability_array.shape[1]} columns for "
                f"{n_classes} classes."
            )

    if probability_array.ndim == 2:
        row_sums = probability_array.sum(axis=1, dtype=float)
        valid_rows = np.isclose(row_sums, 1.0, rtol=1e-7, atol=1e-8)
        if not bool(np.all(valid_rows)):
            first_row = int(np.flatnonzero(~valid_rows)[0])
            raise ValueError(
                "Each row of a y_proba matrix must sum approximately to 1; "
                f"row {first_row} sums to {row_sums[first_row]:.12g}."
            )

    return probability_array


def validate_binary_configuration(
    classes: Sequence[Any],
    positive_label: Any,
) -> int:
    """Validate a binary class configuration and return its positive index.

    Args:
        classes: Ordered class labels; exactly two are required.
        positive_label: Explicitly declared positive label.

    Returns:
        The zero-based index of ``positive_label`` in ``classes``.

    Raises:
        ValueError: If there are not exactly two classes or the positive label
            is missing/invalid.
    """

    class_array = validate_classes(classes)
    if class_array.size != 2:
        raise ValueError(
            "Binary classification requires exactly two classes; "
            f"received {class_array.size}."
        )
    if positive_label is None:
        raise ValueError("positive_label is required for binary classification.")

    positive_array = np.asarray([positive_label], dtype=object)
    _validate_label_values(positive_array, "positive_label")
    for index, label in enumerate(class_array):
        if _labels_equal(label, positive_label):
            return index

    raise ValueError(
        f"positive_label {positive_label!r} is not present in classes "
        f"{class_array.tolist()!r}."
    )


def _resolve_task_type(task_type: str | None, n_classes: int) -> str:
    """Resolve and cross-check an optional task type against class count."""

    if n_classes < 2:
        raise ValueError(
            "Classification evaluation requires at least two declared classes; "
            f"received {n_classes}."
        )

    if task_type is None:
        return "binary" if n_classes == 2 else "multiclass"
    if task_type not in _VALID_TASK_TYPES:
        raise ValueError(
            "task_type must be None, 'binary', or 'multiclass'; "
            f"received {task_type!r}."
        )
    if task_type == "binary" and n_classes != 2:
        raise ValueError(
            "task_type='binary' requires exactly two classes; "
            f"received {n_classes}."
        )
    if task_type == "multiclass" and n_classes < 3:
        raise ValueError(
            "task_type='multiclass' requires at least three classes; "
            f"received {n_classes}."
        )
    return task_type


def validate_evaluation_inputs(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: Sequence[Any] | np.ndarray,
    classes: Sequence[Any],
    class_names: Sequence[str] | None = None,
    positive_label: Any = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Validate the complete classification-evaluation hand-off contract.

    ``task_type`` is inferred from the class count when omitted.  A binary
    configuration always requires an explicit ``positive_label``.  A one-
    dimensional binary ``y_proba`` is interpreted as the probability of that
    declared positive label.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_proba: Binary or multiclass prediction probabilities.
        classes: Ordered class labels matching probability columns.
        class_names: Optional display names in the same order as ``classes``.
        positive_label: Required positive class for binary evaluation.
        task_type: ``None``, ``"binary"``, or ``"multiclass"``.

    Returns:
        Metadata containing normalized arrays and the resolved configuration.

    Raises:
        ValueError: If any part of the hand-off contract is invalid.
    """

    true_array, predicted_array = validate_label_arrays(y_true, y_pred)
    class_array = validate_classes(classes, true_array, predicted_array)
    name_array = validate_class_names(class_names, class_array)

    n_samples = int(true_array.size)
    n_classes = int(class_array.size)
    resolved_task_type = _resolve_task_type(task_type, n_classes)
    probability_array = validate_probability_array(
        y_proba,
        n_samples=n_samples,
        n_classes=n_classes,
        task_type=resolved_task_type,
    )

    positive_index: int | None = None
    if resolved_task_type == "binary":
        positive_index = validate_binary_configuration(class_array, positive_label)
    elif positive_label is not None:
        raise ValueError("positive_label is only valid for binary classification.")

    return {
        "n_samples": n_samples,
        "n_classes": n_classes,
        "task_type": resolved_task_type,
        "classes": class_array,
        "class_names": name_array,
        "positive_label": positive_label,
        "positive_index": positive_index,
        "y_true": true_array,
        "y_pred": predicted_array,
        "y_proba": probability_array,
    }


__all__ = [
    "validate_binary_configuration",
    "validate_classes",
    "validate_class_names",
    "validate_evaluation_inputs",
    "validate_label_arrays",
    "validate_probability_array",
]
