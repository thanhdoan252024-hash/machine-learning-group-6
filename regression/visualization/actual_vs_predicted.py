"""Biểu đồ Actual vs Predicted cho một tập dữ liệu (train hoặc test)."""

from __future__ import annotations

import os
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from regression.utils.validation import validate_regression_arrays


def plot_actual_vs_predicted(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    title: str = "Actual vs Predicted",
    output_path: str | None = None,
) -> str:
    """Vẽ scatter Actual vs Predicted kèm đường chéo lý tưởng y = x.

    Args:
        y_true: Giá trị thực tế.
        y_pred: Giá trị dự đoán.
        title: Tiêu đề biểu đồ.
        output_path: Đường dẫn file PNG lưu ảnh. Nếu ``None`` chỉ hiển thị
            mà không lưu.

    Returns:
        Đường dẫn file đã lưu (hoặc chuỗi rỗng nếu không lưu).

    Raises:
        ValueError: Nếu dữ liệu đầu vào không hợp lệ.
    """

    true_array, predicted_array = validate_regression_arrays(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(true_array, predicted_array, alpha=0.6, edgecolors="k", linewidths=0.5)

    lower = min(float(true_array.min()), float(predicted_array.min()))
    upper = max(float(true_array.max()), float(predicted_array.max()))
    ax.plot([lower, upper], [lower, upper], "r--", linewidth=1.5, label="Lý tưởng (y = x)")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Actual (y_true)", fontsize=11)
    ax.set_ylabel("Predicted (y_pred)", fontsize=11)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    return _save_or_show(fig, output_path)


def _save_or_show(fig, output_path: str | None) -> str:
    """Lưu figure ra PNG nếu có đường dẫn, ngược lại hiển thị."""

    if output_path is None:
        fig.tight_layout()
        fig.show()
        return ""

    resolved = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    fig.tight_layout()
    fig.savefig(resolved, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu ảnh: {resolved}")
    return resolved
