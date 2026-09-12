"""Biểu đồ Feature Importance.

Chỉ vẽ khi model thật cung cấp được độ quan trọng của feature (ví dụ
:class:`regression.lightgbm_regression.LightGBMRegression` có phương thức
``get_feature_importance()``). Không tự sinh dữ liệu giả.
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from regression.visualization.actual_vs_predicted import _save_or_show


def plot_feature_importance(
    importances: Sequence[float],
    feature_names: Sequence[str],
    title: str = "Feature Importance",
    output_path: str | None = None,
) -> str:
    """Vẽ biểu đồ cột ngang thể hiện độ quan trọng của từng feature.

    Args:
        importances: Mảng độ quan trọng của các feature (từ model thật).
        feature_names: Danh sách tên feature, cùng thứ tự với ``importances``.
        title: Tiêu đề biểu đồ.
        output_path: Đường dẫn file PNG lưu ảnh. Nếu ``None`` chỉ hiển thị.

    Returns:
        Đường dẫn file đã lưu (hoặc chuỗi rỗng nếu không lưu).

    Raises:
        ValueError: Nếu ``importances`` rỗng hoặc độ dài không khớp
            ``feature_names``.
    """

    importance_array = np.asarray(importances, dtype=float).reshape(-1)
    names = list(feature_names)

    if importance_array.size == 0:
        raise ValueError("importances không được rỗng; không vẽ feature importance.")
    if importance_array.size != len(names):
        raise ValueError(
            "importances và feature_names phải cùng độ dài; "
            f"nhận {importance_array.size} và {len(names)}."
        )

    # Sắp xếp giảm dần theo độ quan trọng.
    order = np.argsort(importance_array)[::-1]
    sorted_importance = importance_array[order]
    sorted_names = [names[index] for index in order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(sorted_names))))
    ax.barh(sorted_names, sorted_importance, color="steelblue")
    ax.invert_yaxis()

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Độ quan trọng", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)

    return _save_or_show(fig, output_path)
