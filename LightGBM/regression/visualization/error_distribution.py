"""Error Distribution (phân phối sai số) cho một tập dữ liệu (train hoặc test)."""

from __future__ import annotations

import os
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from regression.utils.validation import validate_regression_arrays
from regression.visualization.actual_vs_predicted import _save_or_show


def plot_error_distribution(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    title: str = "Error Distribution",
    output_path: str | None = None,
) -> str:
    """Vẽ histogram + KDE của sai số (error = y_true - y_pred).

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
    errors = true_array - predicted_array

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(errors, kde=True, ax=ax, color="steelblue", edgecolor="white")
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="Error = 0")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Error (y_true - y_pred)", fontsize=11)
    ax.set_ylabel("Tần suất", fontsize=11)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    return _save_or_show(fig, output_path)
