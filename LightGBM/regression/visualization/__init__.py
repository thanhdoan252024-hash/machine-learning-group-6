"""Package visualization regression: vẽ biểu đồ cho tập train và test.

Tất cả hàm nhận ``output_path`` để lưu PNG vào ``regression/outputs/figures/``.
"""

from regression.visualization.actual_vs_predicted import plot_actual_vs_predicted
from regression.visualization.residual_plot import plot_residuals
from regression.visualization.error_distribution import plot_error_distribution
from regression.visualization.train_test_comparison import (
    plot_train_test_metrics_comparison,
)
from regression.visualization.feature_importance import plot_feature_importance

__all__ = [
    "plot_actual_vs_predicted",
    "plot_residuals",
    "plot_error_distribution",
    "plot_train_test_metrics_comparison",
    "plot_feature_importance",
]
