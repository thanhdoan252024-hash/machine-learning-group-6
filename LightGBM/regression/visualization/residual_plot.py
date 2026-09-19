"""Residual Plot cho một tập dữ liệu (train hoặc test)."""

from __future__ import annotations

import os
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from regression.utils.validation import validate_regression_arrays
from regression.visualization.actual_vs_predicted import _save_or_show


def plot_residuals(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    title: str = "Residual Plot",
    output_path: str | None = None,
) -> str:
    """Vẽ residual (y_true - y_pred) theo predicted value, kèm đường zero.

    Args:
        y_true: Giá trị thực tế.
        y_pred: Giá trị dự đoán.
        title: Tiêu đề biểu đồ.
        output_path: Đường dẫn file PNG lưu ảnh. Nếu ``None`` chỉ hiển thị.

    Returns:
        Đường dẫn file đã lưu (hoặc chuỗi rỗng nếu không lưu).

    Raises:
        ValueError: Nếu dữ liệu đầu vào không hợp lệ.
    """

    true_array, predicted_array = validate_regression_arrays(y_true, y_pred)
    residuals = true_array - predicted_array

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predicted_array, residuals, alpha=0.6, edgecolors="k", linewidths=0.5)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.5, label="Residual = 0")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted (y_pred)", fontsize=11)
    ax.set_ylabel("Residual (y_true - y_pred)", fontsize=11)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    return _save_or_show(fig, output_path)
