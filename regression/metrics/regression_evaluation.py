"""Đánh giá tự động regression cho cả tập train và test.

File tổng hợp này gọi ba metric (MAE, MSE, R2) cho từng tập dữ liệu, in kết
quả ra console và lưu vào ``outputs/result/train_test_result.csv``.

Cách dùng trong notebook:

.. code-block:: python

    from regression.metrics import evaluate_regression

    evaluate_regression(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
    )
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

from regression.metrics.mean_absolute_error import mean_absolute_error
from regression.metrics.mean_squared_error import mean_squared_error
from regression.metrics.r2_score import r2_score

# Đường dẫn mặc định tương đối với thư mục chứa package regression.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUTPUT_PATH = os.path.join(
    _HERE, "..", "outputs", "result", "train_test_result.csv"
)

_METRIC_FUNCTIONS = {
    "MAE": mean_absolute_error,
    "MSE": mean_squared_error,
    "R2": r2_score,
}


def _compute_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> dict[str, float]:
    """Tính cả ba metric cho một cặp ``(y_true, y_pred)``."""

    return {name: func(y_true, y_pred) for name, func in _METRIC_FUNCTIONS.items()}


def _format(value: float) -> str:
    """Định dạng số thập phân để in và ghi CSV nhất quán."""

    return f"{value:.6f}"


def evaluate_regression(
    y_train: Sequence[Any],
    y_train_pred: Sequence[Any],
    y_test: Sequence[Any],
    y_test_pred: Sequence[Any],
    output_path: str | None = None,
) -> dict[str, dict[str, float]]:
    """Đánh giá tự động mô hình regression trên tập train và test.

    Args:
        y_train: Giá trị thực tế của tập train.
        y_train_pred: Giá trị dự đoán của tập train.
        y_test: Giá trị thực tế của tập test.
        y_test_pred: Giá trị dự đoán của tập test.
        output_path: Đường dẫn file CSV lưu kết quả. Mặc định là
            ``regression/outputs/result/train_test_result.csv``.

    Returns:
        Dict dạng ``{"train": {"MAE": ..., "MSE": ..., "R2": ...},
        "test": {...}}`` chứa kết quả từng metric.

    Raises:
        ValueError: Nếu bất kỳ cặp ``(y_true, y_pred)`` nào không hợp lệ.
    """

    results = {
        "train": _compute_metrics(y_train, y_train_pred),
        "test": _compute_metrics(y_test, y_test_pred),
    }

    # In kết quả ra console.
    print("=" * 56)
    print("ĐÁNH GIÁ REGRESSION - TRAIN & TEST")
    print("=" * 56)
    header = f"{'Split':<8}{'MAE':>12}{'MSE':>14}{'R2':>12}"
    print(header)
    print("-" * 56)
    for split in ("train", "test"):
        metrics = results[split]
        print(
            f"{split:<8}"
            f"{_format(metrics['MAE']):>12}"
            f"{_format(metrics['MSE']):>14}"
            f"{_format(metrics['R2']):>12}"
        )
    print("=" * 56)

    # Lưu kết quả ra CSV.
    resolved_path = os.path.abspath(output_path or _DEFAULT_OUTPUT_PATH)
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
    _write_csv(resolved_path, results)

    print(f"Đã lưu kết quả vào: {resolved_path}")
    return results


def _write_csv(path: str, results: dict[str, dict[str, float]]) -> None:
    """Ghi kết quả dạng bảng vào file CSV (không dùng pandas)."""

    metric_names = list(_METRIC_FUNCTIONS.keys())
    lines = ["split," + ",".join(metric_names)]
    for split in ("train", "test"):
        row = [split] + [_format(results[split][name]) for name in metric_names]
        lines.append(",".join(row))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def load_train_test_result(path: str | None = None) -> np.ndarray:
    """Đọc lại file CSV kết quả thành mảng NumPy (tiện cho notebook).

    Args:
        path: Đường dẫn file CSV. Mặc định dùng file mặc định của package.

    Returns:
        Mảng NumPy 2 chiều chứa nội dung bảng kết quả (kèm header).
    """

    resolved_path = os.path.abspath(path or _DEFAULT_OUTPUT_PATH)
    with open(resolved_path, "r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    return np.array([line.split(",") for line in lines])
