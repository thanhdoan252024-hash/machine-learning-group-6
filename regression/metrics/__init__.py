"""Package metrics regression: MAE, MSE, R2 và đánh giá tự động train/test.

Tất cả metric được tính bằng NumPy thuần, không dùng ``sklearn.metrics``.
"""

from regression.metrics.mean_absolute_error import mean_absolute_error
from regression.metrics.mean_squared_error import mean_squared_error
from regression.metrics.r2_score import r2_score
from regression.metrics.regression_evaluation import (
    evaluate_regression,
    load_train_test_result,
)

__all__ = [
    "mean_absolute_error",
    "mean_squared_error",
    "r2_score",
    "evaluate_regression",
    "load_train_test_result",
]
