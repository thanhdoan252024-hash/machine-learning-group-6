"""Tạo bảng classification report từ các metric đã được tính thủ công."""

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPORT_COLUMNS = [
    "label",
    "row_type",
    "precision",
    "recall",
    "f1_score",
    "support",
]


def create_classification_report_dataframe(
    per_class_metrics: Sequence[Mapping[str, Any]],
    aggregate_metrics: Mapping[str, float],
    accuracy: float,
) -> pd.DataFrame:
    """Tổ chức metric có sẵn thành một classification report.

    Hàm này không tính lại Precision, Recall hoặc F1. Dòng ``accuracy`` đặt
    Accuracy trong cột ``f1_score`` và để trống Precision/Recall, tương tự cách
    biểu diễn một metric tổng thể trong bảng có schema cố định.

    Args:
        per_class_metrics: Danh sách kết quả từng lớp từ
            :func:`evaluation.manual_metrics.calculate_per_class_metrics`.
        aggregate_metrics: Dictionary chứa macro và weighted metrics.
        accuracy: Accuracy tổng thể đã được tính thủ công.

    Returns:
        DataFrame có một dòng cho mỗi lớp, accuracy, macro avg và weighted avg.

    Raises:
        ValueError: Nếu dữ liệu metric thiếu trường hoặc chứa giá trị không hợp lệ.
    """
    if not per_class_metrics:
        raise ValueError("per_class_metrics không được rỗng.")

    accuracy_value = _validate_unit_interval(accuracy, "accuracy")
    required_aggregate_keys = (
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    )
    missing_aggregate = [
        key for key in required_aggregate_keys if key not in aggregate_metrics
    ]
    if missing_aggregate:
        raise ValueError(
            "aggregate_metrics thiếu các trường: " + ", ".join(missing_aggregate)
        )

    rows: list[dict[str, Any]] = []
    total_support = 0
    for index, item in enumerate(per_class_metrics):
        required_class_keys = ("precision", "recall", "f1_score", "support")
        missing_class = [key for key in required_class_keys if key not in item]
        if missing_class:
            raise ValueError(
                f"per_class_metrics[{index}] thiếu các trường: "
                + ", ".join(missing_class)
            )

        support = _validate_support(item["support"], f"support của lớp {index}")
        total_support += support
        label = item.get("class_name", item.get("class_label", f"Class {index}"))
        label_text = str(label).strip()
        if not label_text:
            raise ValueError(f"Tên hiển thị của lớp tại vị trí {index} không được rỗng.")

        rows.append(
            {
                "label": label_text,
                "row_type": "class",
                "precision": _validate_unit_interval(
                    item["precision"], f"precision của lớp {label_text}"
                ),
                "recall": _validate_unit_interval(
                    item["recall"], f"recall của lớp {label_text}"
                ),
                "f1_score": _validate_unit_interval(
                    item["f1_score"], f"f1_score của lớp {label_text}"
                ),
                "support": support,
            }
        )

    rows.extend(
        [
            {
                "label": "accuracy",
                "row_type": "summary",
                "precision": np.nan,
                "recall": np.nan,
                "f1_score": accuracy_value,
                "support": total_support,
            },
            {
                "label": "macro avg",
                "row_type": "summary",
                "precision": _validate_unit_interval(
                    aggregate_metrics["precision_macro"], "precision_macro"
                ),
                "recall": _validate_unit_interval(
                    aggregate_metrics["recall_macro"], "recall_macro"
                ),
                "f1_score": _validate_unit_interval(
                    aggregate_metrics["f1_macro"], "f1_macro"
                ),
                "support": total_support,
            },
            {
                "label": "weighted avg",
                "row_type": "summary",
                "precision": _validate_unit_interval(
                    aggregate_metrics["precision_weighted"],
                    "precision_weighted",
                ),
                "recall": _validate_unit_interval(
                    aggregate_metrics["recall_weighted"], "recall_weighted"
                ),
                "f1_score": _validate_unit_interval(
                    aggregate_metrics["f1_weighted"], "f1_weighted"
                ),
                "support": total_support,
            },
        ]
    )

    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def _validate_unit_interval(value: Any, name: str) -> float:
    """Chuyển và kiểm tra một metric thuộc đoạn [0, 1]."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} phải là một số thực.") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} phải là một số hữu hạn.")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} phải nằm trong đoạn [0, 1], nhận được {numeric}.")
    return numeric


def _validate_support(value: Any, name: str) -> int:
    """Kiểm tra support là số nguyên không âm."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} phải là số nguyên không âm.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} phải là số nguyên không âm.") from exc
    if not np.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError(f"{name} phải là số nguyên không âm.")
    return int(numeric)
