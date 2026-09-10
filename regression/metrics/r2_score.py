"""R-squared (R2) score tính bằng NumPy thuần, không dùng sklearn."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from regression.utils.validation import validate_regression_arrays


def r2_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> float:
    """Tính hệ số xác định R-squared (R2).

    .. math::

        R^2 = 1 - \\frac{\\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}
                       {\\sum_{i=1}^{n} (y_i - \\bar{y})^2}

    trong đó :math:`\\bar{y}` là trung bình của ``y_true``.

    Args:
        y_true: Giá trị thực tế.
        y_pred: Giá trị dự đoán.

    Returns:
        Giá trị R2 dạng float.

    Raises:
        ValueError: Nếu dữ liệu đầu vào không hợp lệ (rỗng, khác độ dài,
            không phải mảng một chiều, chứa NaN/infinity).
    """

    true_array, predicted_array = validate_regression_arrays(y_true, y_pred)
    residual_sum_of_squares = np.sum((true_array - predicted_array) ** 2)
    total_sum_of_squares = np.sum((true_array - np.mean(true_array)) ** 2)

    if total_sum_of_squares == 0:
        # Mọi giá trị y_true đều bằng nhau nên mẫu số bằng 0; khi đó mô hình
        # chỉ "đúng" nếu dự đoán khớp hoàn toàn, ngược lại R2 coi như không
        # xác định và ta quy ước bằng 0.0.
        if residual_sum_of_squares == 0:
            return 1.0
        return 0.0

    return float(1.0 - residual_sum_of_squares / total_sum_of_squares)
