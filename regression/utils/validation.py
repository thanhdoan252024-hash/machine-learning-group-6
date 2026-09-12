"""Validation helpers cho các metric regression.

Module này kiểm tra điều kiện dữ liệu đầu vào ``y_true`` và ``y_pred`` trước khi
tính toán. Các metric chỉ nhận dữ liệu đã được validate để tránh kết quả sai
hoặc lỗi khó hiểu.

Nguyên tắc:
- Không được rỗng.
- Phải cùng số phần tử.
- Phải là mảng một chiều chứa giá trị số hữu hạn (không NaN, không infinity).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _as_float_array(values: Sequence[Any], name: str) -> np.ndarray:
    """Chuyển *values* thành mảng NumPy một chiều kiểu float."""

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} phải chuyển được thành mảng số; gặp {type(values).__name__}."
        ) from exc

    if array.ndim != 1:
        raise ValueError(
            f"{name} phải là mảng một chiều; nhận mảng {array.ndim} chiều "
            f"với shape {array.shape}."
        )
    return array


def validate_regression_arrays(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate và trả về cặp ``(y_true, y_pred)`` dạng mảng float một chiều.

    Args:
        y_true: Giá trị thực tế (ground-truth).
        y_pred: Giá trị dự đoán.

    Returns:
        Tuple ``(validated_y_true, validated_y_pred)`` là mảng NumPy float
        một chiều, cùng độ dài.

    Raises:
        ValueError: Nếu một trong hai mảng rỗng, khác số phần tử, không phải
            mảng một chiều, hoặc chứa NaN / infinity.
    """

    true_array = _as_float_array(y_true, "y_true")
    predicted_array = _as_float_array(y_pred, "y_pred")

    if true_array.size == 0:
        raise ValueError("y_true không được rỗng.")
    if predicted_array.size == 0:
        raise ValueError("y_pred không được rỗng.")
    if true_array.size != predicted_array.size:
        raise ValueError(
            "y_true và y_pred phải có cùng số phần tử; "
            f"nhận {true_array.size} và {predicted_array.size}."
        )

    for array, name in ((true_array, "y_true"), (predicted_array, "y_pred")):
        if np.any(np.isnan(array)):
            raise ValueError(f"{name} chứa giá trị NaN.")
        if np.any(np.isinf(array)):
            raise ValueError(f"{name} chứa giá trị vô cùng (infinity).")

    return true_array, predicted_array
