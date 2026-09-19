"""Biểu đồ cột so sánh 3 metric (MAE, MSE, R2) giữa Train và Test."""

from __future__ import annotations

import os
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from regression.visualization.actual_vs_predicted import _save_or_show


def plot_train_test_metrics_comparison(
    train_metrics: Mapping[str, float],
    test_metrics: Mapping[str, float],
    title: str = "So sánh Metrics Train vs Test",
    output_path: str | None = None,
) -> str:
    """Vẽ biểu đồ cột nhóm so sánh trực tiếp MAE, MSE, R2 giữa Train và Test.

    Mỗi metric có hai cột đặt cạnh nhau (Train và Test) để dễ phát hiện
    overfitting (test kém hơn train đáng kể).

    Args:
        train_metrics: Dict chứa metric của tập train (MAE, MSE, R2).
        test_metrics: Dict chứa metric của tập test (MAE, MSE, R2).
        title: Tiêu đề biểu đồ.
        output_path: Đường dẫn file PNG lưu ảnh. Nếu ``None`` chỉ hiển thị.

    Returns:
        Đường dẫn file đã lưu (hoặc chuỗi rỗng nếu không lưu).
    """

    metric_names = ["MAE", "MSE", "R2"]
    train_values = [float(train_metrics[name]) for name in metric_names]
    test_values = [float(test_metrics[name]) for name in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars_train = ax.bar(x - width / 2, train_values, width, label="Train", color="steelblue")
    bars_test = ax.bar(x + width / 2, test_values, width, label="Test", color="darkorange")

    ax.set_title(title, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel("Giá trị metric", fontsize=11)
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Ghi giá trị lên đầu mỗi cột.
    for bars in (bars_train, bars_test):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    return _save_or_show(fig, output_path)
