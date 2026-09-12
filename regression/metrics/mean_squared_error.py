"""Mean Squared Error (MSE) tính bằng NumPy thuần, không dùng sklearn."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from regression.utils.validation import validate_regression_arrays


def mean_squared_error(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> float:
    """Tính Mean Squared Error (MSE).

    .. math::

        MSE = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2

    Args:
        y_true: Giá trị thực tế.
        y_pred: Giá trị dự đoán.

    Returns:
        Giá trị MSE dạng float.

    Raises:
        ValueError: Nếu dữ liệu đầu vào không hợp lệ (rỗng, khác độ dài,
            không phải mảng một chiều, chứa NaN/infinity).
    """

    true_array, predicted_array = validate_regression_arrays(y_true, y_pred)
    return float(np.mean((true_array - predicted_array) ** 2))
