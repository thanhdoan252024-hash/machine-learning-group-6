"""Luồng đánh giá tự động kết hợp metrics + CSV + toàn bộ biểu đồ.

Đây là điểm vào duy nhất: chỉ cần truyền kết quả dự đoán (train và test),
hàm sẽ tự tính 3 metric (MAE, MSE, R2), in kết quả, lưu CSV vào
``outputs/result/train_test_result.csv`` và xuất toàn bộ 8 ảnh vào
``outputs/figures/``.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from regression.metrics.regression_evaluation import evaluate_regression
from regression.visualization.actual_vs_predicted import plot_actual_vs_predicted
from regression.visualization.residual_plot import plot_residuals
from regression.visualization.error_distribution import plot_error_distribution
from regression.visualization.train_test_comparison import (
    plot_train_test_metrics_comparison,
)
from regression.visualization.feature_importance import plot_feature_importance

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_FIGURES_DIR = os.path.join(_HERE, "outputs", "figures")
_DEFAULT_RESULT_CSV = os.path.join(_HERE, "outputs", "result", "train_test_result.csv")


def _figure_path(figures_dir: str, filename: str) -> str:
    return os.path.join(figures_dir, filename)


def evaluate_and_visualize(
    y_train: Sequence[Any],
    y_train_pred: Sequence[Any],
    y_test: Sequence[Any],
    y_test_pred: Sequence[Any],
    model: Any = None,
    feature_names: Sequence[str] | None = None,
    figures_dir: str | None = None,
    result_csv_path: str | None = None,
) -> dict[str, dict[str, float]]:
    """Đánh giá tự động và xuất toàn bộ biểu đồ cho tập train và test.

    Args:
        y_train: Giá trị thực tế của tập train.
        y_train_pred: Giá trị dự đoán của tập train.
        y_test: Giá trị thực tế của tập test.
        y_test_pred: Giá trị dự đoán của tập test.
        model: Model đã huấn luyện. Nếu có phương thức ``get_feature_importance()``
            và ``feature_names`` được cung cấp thì sẽ vẽ thêm biểu đồ feature
            importance. Không tự sinh dữ liệu giả.
        feature_names: Danh sách tên feature (cùng thứ tự với dữ liệu huấn luyện).
        figures_dir: Thư mục lưu ảnh. Mặc định ``regression/outputs/figures/``.
        result_csv_path: Đường dẫn file CSV lưu kết quả metrics.

    Returns:
        Dict kết quả metrics dạng ``{"train": {...}, "test": {...}}``.

    Raises:
        ValueError: Nếu bất kỳ cặp ``(y_true, y_pred)`` nào không hợp lệ.
    """

    # 1) Tính metrics, in kết quả và lưu CSV.
    results = evaluate_regression(
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        output_path=result_csv_path or _DEFAULT_RESULT_CSV,
    )

    # 2) Xuất toàn bộ biểu đồ.
    figures_dir = os.path.abspath(figures_dir or _DEFAULT_FIGURES_DIR)
    os.makedirs(figures_dir, exist_ok=True)

    plot_actual_vs_predicted(
        y_train,
        y_train_pred,
        title="Train: Actual vs Predicted",
        output_path=_figure_path(figures_dir, "train_actual_vs_predicted.png"),
    )
    plot_actual_vs_predicted(
        y_test,
        y_test_pred,
        title="Test: Actual vs Predicted",
        output_path=_figure_path(figures_dir, "test_actual_vs_predicted.png"),
    )

    plot_residuals(
        y_train,
        y_train_pred,
        title="Train: Residual Plot",
        output_path=_figure_path(figures_dir, "train_residual_plot.png"),
    )
    plot_residuals(
        y_test,
        y_test_pred,
        title="Test: Residual Plot",
        output_path=_figure_path(figures_dir, "test_residual_plot.png"),
    )

    plot_error_distribution(
        y_train,
        y_train_pred,
        title="Train: Error Distribution",
        output_path=_figure_path(figures_dir, "train_error_distribution.png"),
    )
    plot_error_distribution(
        y_test,
        y_test_pred,
        title="Test: Error Distribution",
        output_path=_figure_path(figures_dir, "test_error_distribution.png"),
    )

    plot_train_test_metrics_comparison(
        results["train"],
        results["test"],
        title="So sánh Metrics Train vs Test",
        output_path=_figure_path(figures_dir, "train_test_metrics_bar_chart.png"),
    )

    # 3) Feature importance — chỉ vẽ nếu model thật hỗ trợ.
    importances = _extract_feature_importances(model)
    if importances is not None and feature_names is not None:
        try:
            plot_feature_importance(
                importances,
                feature_names,
                title="Feature Importance",
                output_path=_figure_path(figures_dir, "feature_importance.png"),
            )
        except ValueError as exc:
            print(f"Bỏ qua feature importance: {exc}")
    else:
        print(
            "Bỏ qua feature importance: model không cung cấp get_feature_importance() "
            "hoặc chưa truyền feature_names."
        )

    print(f"Đã xuất toàn bộ ảnh vào: {figures_dir}")
    return results


def _extract_feature_importances(model: Any) -> Sequence[float] | None:
    """Lấy độ quan trọng feature từ model nếu có hỗ trợ."""

    if model is None:
        return None
    getter = getattr(model, "get_feature_importance", None)
    if not callable(getter):
        return None
    try:
        importances = getter()
    except Exception:
        return None
    if importances is None:
        return None
    return list(importances)
