"""Kết nối output của ``LightGBMClassification`` với bộ đánh giá độc lập.

Adapter này chỉ chuẩn hóa metadata và chuyển các mảng dự đoán sang runner dùng
chung. Toàn bộ công thức metric, ROC-AUC, bảng và biểu đồ vẫn thuộc các module
``evaluation``/``experiments``; không có metric nào được tính lại tại đây.
"""

from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from experiments.run_classification_evaluation import (
    run_classification_evaluation,
)


POSITIVE_LABEL = 1
"""Nhãn biểu thị máy hỏng trong cột ``Machine failure`` của dataset."""

CLASS_NAME_BY_LABEL = {
    0: "Không hỏng máy",
    1: "Hỏng máy",
}
"""Tên hiển thị ổn định, được ánh xạ theo giá trị nhãn thay vì vị trí."""


class FittedBinaryClassifier(Protocol):
    """Giao diện tối thiểu mà adapter cần từ model đã huấn luyện."""

    classes_: Sequence[Any]

    def predict(self, X: Any) -> Any:
        """Trả nhãn dự đoán cho từng mẫu."""

    def predict_proba(self, X: Any) -> Any:
        """Trả xác suất theo đúng thứ tự cột trong ``classes_``."""


def evaluate_classification_outputs(
    model: FittedBinaryClassifier,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: Any,
    output_dir: str | Path,
    *,
    save_dpi: int = 300,
) -> dict[str, Any]:
    """Đánh giá các output đã được model tạo ra.

    ``classes`` không phải tham số đầu vào có thể cấu hình tùy ý: adapter luôn
    đọc trực tiếp ``model.classes_`` để giữ đúng thứ tự cột của
    ``predict_proba``. Với bài toán machine-failure hiện tại, lớp dương phải là
    nhãn số ``1`` và phải ở cột xác suất thứ hai, đúng với cách
    :meth:`LightGBMClassification.predict` áp dụng threshold.

    Args:
        model: Model binary đã fit và có thuộc tính ``classes_``.
        y_true: Nhãn thật một chiều.
        y_pred: Nhãn dự đoán một chiều do ``model.predict`` trả về.
        y_proba: Ma trận xác suất do ``model.predict_proba`` trả về.
        output_dir: Thư mục output; convention của repo là
            ``<repo>/classification/outputs``.
        save_dpi: Độ phân giải của các hình PNG.

    Returns:
        Kết quả đầy đủ từ ``run_classification_evaluation``.

    Raises:
        ValueError: Nếu model chưa fit hoặc metadata lớp không đúng hợp đồng
            binary machine-failure ``[0, 1]``.
    """

    classes, class_names = _resolve_model_class_metadata(model)
    return run_classification_evaluation(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=classes,
        class_names=class_names,
        positive_label=POSITIVE_LABEL,
        task_type="binary",
        output_dir=output_dir,
        save_dpi=save_dpi,
    )


def evaluate_fitted_classifier(
    model: FittedBinaryClassifier,
    X: Any,
    y_true: Sequence[Any],
    output_dir: str | Path,
    *,
    save_dpi: int = 300,
) -> dict[str, Any]:
    """Tạo prediction từ model đã fit rồi chạy đánh giá đầy đủ.

    Hàm này phù hợp với script chạy end-to-end. Notebook đã có sẵn
    ``predict_test`` và ``predict_test_proba`` nên nên dùng
    :func:`evaluate_classification_outputs` để không dự đoán lặp lại.
    """

    # Kiểm tra metadata trước để báo lỗi cấu hình rõ ràng, thay vì gọi predict
    # trên một model chưa fit hoặc có thứ tự lớp không phù hợp.
    _resolve_model_class_metadata(model)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return evaluate_classification_outputs(
        model=model,
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        output_dir=output_dir,
        save_dpi=save_dpi,
    )


def _resolve_model_class_metadata(
    model: FittedBinaryClassifier,
) -> tuple[list[Any], list[str]]:
    """Đọc và kiểm tra classes từ model theo hợp đồng dataset hiện tại."""

    if not hasattr(model, "classes_"):
        raise ValueError("Model phải được fit và cung cấp thuộc tính classes_.")

    raw_classes = getattr(model, "classes_")
    try:
        class_array = np.asarray(raw_classes, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("model.classes_ phải là một danh sách nhãn hợp lệ.") from exc

    if class_array.ndim != 1 or class_array.size != 2:
        raise ValueError(
            "model.classes_ phải chứa đúng hai nhãn cho bài toán binary; "
            f"nhận shape {class_array.shape}."
        )

    classes = class_array.tolist()
    if classes != [0, POSITIVE_LABEL]:
        raise ValueError(
            "Bài toán Machine failure yêu cầu model.classes_ theo đúng thứ tự "
            f"[0, {POSITIVE_LABEL}]; nhận {classes!r}."
        )

    # Ánh xạ bằng giá trị label để tên vẫn gắn với đúng semantic class.
    class_names = [CLASS_NAME_BY_LABEL[label] for label in classes]
    return classes, class_names


__all__ = [
    "CLASS_NAME_BY_LABEL",
    "POSITIVE_LABEL",
    "evaluate_classification_outputs",
    "evaluate_fitted_classifier",
]
