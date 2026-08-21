"""Điều phối đánh giá classification từ các mảng dự đoán đã có.

Module này cố ý không import model hoặc dataset. Adapter dành cho bài toán hỏng
máy nằm tại ``classification.classification_evaluation_adapter`` và chỉ chuẩn
hóa output trước khi gọi ``run_classification_evaluation``.
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence


# Cho phép chạy trực tiếp `python experiments/run_classification_evaluation.py`
# trước khi project được đóng gói/cài đặt. Khi import như module, không đổi sys.path.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from evaluation.exporters import export_evaluation_results
from evaluation.input_validation import validate_evaluation_inputs
from evaluation.manual_metrics import evaluate_classification
from evaluation.manual_roc_auc import (
    calculate_binary_roc_curve,
    calculate_multiclass_roc_ovr,
    extract_positive_scores,
)
from evaluation.reports import create_classification_report_dataframe
from evaluation.visualizations import (
    plot_binary_roc_curve,
    plot_confusion_matrix,
    plot_correct_incorrect_pie,
    plot_multiclass_roc_curves,
    plot_overall_metrics_bar,
    plot_per_class_metrics,
)


def run_classification_evaluation(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: Any,
    classes: Sequence[Any],
    output_dir: str | Path,
    class_names: Sequence[str] | None = None,
    positive_label: Any | None = None,
    task_type: str | None = None,
    save_dpi: int = 300,
) -> dict[str, Any]:
    """Chạy toàn bộ phần đánh giá độc lập từ các mảng kết quả.

    Args:
        y_true: Nhãn thật một chiều.
        y_pred: Nhãn dự đoán một chiều.
        y_proba: Xác suất binary hoặc multiclass.
        classes: Nhãn lớp theo đúng thứ tự cột xác suất.
        output_dir: Thư mục gốc chứa ``tables``, ``figures`` và ``predictions``.
        class_names: Tên hiển thị theo cùng thứ tự với ``classes``. Có thể là
            ``None`` để dùng chuỗi của label.
        positive_label: Bắt buộc và phải chỉ định rõ đối với binary.
        task_type: Có thể là ``binary``/``multiclass`` hoặc ``None`` để suy ra.
        save_dpi: Độ phân giải PNG, mặc định 300 DPI.

    Returns:
        Dictionary chứa metadata, metric, report, ROC và đường dẫn output.

    Raises:
        ValueError: Nếu đầu vào hoặc cấu hình không đáp ứng hợp đồng đánh giá.
    """
    if isinstance(save_dpi, bool) or not isinstance(save_dpi, int) or save_dpi <= 0:
        raise ValueError("save_dpi phải là số nguyên dương.")

    metadata = validate_evaluation_inputs(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=classes,
        class_names=class_names,
        positive_label=positive_label,
        task_type=task_type,
    )

    resolved_true = metadata["y_true"]
    resolved_pred = metadata["y_pred"]
    resolved_proba = metadata["y_proba"]
    resolved_classes = metadata["classes"]
    resolved_names = metadata["class_names"]
    resolved_task = metadata["task_type"]

    metrics = evaluate_classification(
        y_true=resolved_true,
        y_pred=resolved_pred,
        classes=resolved_classes,
        class_names=resolved_names,
        positive_label=positive_label,
    )

    if resolved_task == "binary":
        positive_scores = extract_positive_scores(
            resolved_proba,
            resolved_classes,
            positive_label,
        )
        roc_results = calculate_binary_roc_curve(
            resolved_true,
            positive_scores,
            positive_label,
        )
        metrics["roc_auc"] = float(roc_results["auc"])
        overall_metrics = {
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-score": metrics["f1_score"],
            "ROC-AUC": metrics["roc_auc"],
        }
    else:
        roc_results = calculate_multiclass_roc_ovr(
            resolved_true,
            resolved_proba,
            resolved_classes,
            resolved_names,
        )
        macro_auc = roc_results["macro_auc"]
        metrics["roc_auc_macro"] = (
            None if macro_auc is None else float(macro_auc)
        )
        overall_metrics = {
            "Accuracy": metrics["accuracy"],
            "Macro Precision": metrics["precision_macro"],
            "Macro Recall": metrics["recall_macro"],
            "Macro F1": metrics["f1_macro"],
        }
        if metrics["roc_auc_macro"] is not None:
            overall_metrics["Macro ROC-AUC"] = metrics["roc_auc_macro"]

    report_df = create_classification_report_dataframe(
        per_class_metrics=metrics["per_class_metrics"],
        aggregate_metrics=metrics,
        accuracy=metrics["accuracy"],
    )

    output_root = Path(output_dir)
    figures_dir = output_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: dict[str, Path] = {
        "confusion_matrix_counts": plot_confusion_matrix(
            metrics["confusion_matrix"],
            resolved_names,
            normalized=False,
            output_path=figures_dir / "confusion_matrix_counts.png",
            dpi=save_dpi,
        ),
        "confusion_matrix_normalized": plot_confusion_matrix(
            metrics["normalized_confusion_matrix"],
            resolved_names,
            normalized=True,
            output_path=figures_dir / "confusion_matrix_normalized.png",
            dpi=save_dpi,
        ),
        "overall_metrics_bar": plot_overall_metrics_bar(
            overall_metrics,
            figures_dir / "overall_metrics_bar.png",
            dpi=save_dpi,
        ),
        "per_class_metrics": plot_per_class_metrics(
            report_df,
            figures_dir / "per_class_metrics.png",
            dpi=save_dpi,
        ),
        "correct_incorrect_pie": plot_correct_incorrect_pie(
            resolved_true,
            resolved_pred,
            figures_dir / "correct_incorrect_pie.png",
            dpi=save_dpi,
        ),
    }

    if resolved_task == "binary":
        figure_paths["roc_curve"] = plot_binary_roc_curve(
            roc_results["fpr"],
            roc_results["tpr"],
            float(roc_results["auc"]),
            figures_dir / "roc_curve.png",
            dpi=save_dpi,
        )
    else:
        figure_paths["roc_curve"] = plot_multiclass_roc_curves(
            roc_results,
            resolved_names,
            roc_results["macro_auc"],
            figures_dir / "roc_ovr_multiclass.png",
            dpi=save_dpi,
        )

    table_paths = export_evaluation_results(
        metrics=metrics,
        report_df=report_df,
        confusion_matrix=metrics["confusion_matrix"],
        normalized_confusion_matrix=metrics["normalized_confusion_matrix"],
        y_true=resolved_true,
        y_pred=resolved_pred,
        y_proba=resolved_proba,
        classes=resolved_classes,
        class_names=resolved_names,
        positive_label=positive_label,
        output_dir=output_root,
        roc_results=roc_results,
    )
    manifest_path = _update_output_manifest(
        output_root,
        resolved_task,
        table_paths,
        figure_paths,
    )

    return {
        "metadata": metadata,
        "metrics": metrics,
        "classification_report": report_df,
        "roc_results": roc_results,
        "table_paths": table_paths,
        "figure_paths": figure_paths,
        "manifest_path": manifest_path,
    }


def _update_output_manifest(
    output_root: Path,
    task_type: str,
    table_paths: Mapping[str, Any],
    figure_paths: Mapping[str, Any],
) -> Path:
    """Ghi manifest và chỉ xóa artifact cũ do manifest trước sở hữu.

    Cleanup diễn ra sau khi run mới đã tạo thành công toàn bộ output. Vì vậy lỗi
    validation/render/export không xóa kết quả tốt của lần chạy trước. Manifest
    không dùng glob và không cho phép path thoát khỏi ``output_root``.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "evaluation_manifest.json"
    previous_paths = _read_manifest_paths(manifest_path)
    generated_paths = _collect_output_paths(table_paths, figure_paths)
    current_relative_paths = {
        _relative_output_path(path, output_root) for path in generated_paths
    }

    for stale_relative_path in sorted(previous_paths - current_relative_paths):
        stale_path = _safe_manifest_target(output_root, stale_relative_path)
        if stale_path is None:
            warnings.warn(
                f"Bỏ qua path không an toàn trong output manifest: "
                f"{stale_relative_path!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if stale_path.is_file():
            stale_path.unlink()
        elif stale_path.exists():
            warnings.warn(
                f"Không xóa stale output vì path không phải file: {stale_path}",
                RuntimeWarning,
                stacklevel=2,
            )

    manifest = {
        "schema_version": 1,
        "task_type": task_type,
        "generated_files": sorted(current_relative_paths),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _collect_output_paths(*path_mappings: Mapping[str, Any]) -> set[Path]:
    """Thu thập Path từ các dictionary output lồng nhau."""
    collected: set[Path] = set()

    def collect(value: Any) -> None:
        if isinstance(value, Path):
            collected.add(value)
        elif isinstance(value, Mapping):
            for nested_value in value.values():
                collect(nested_value)

    for path_mapping in path_mappings:
        collect(path_mapping)
    return collected


def _relative_output_path(path: Path, output_root: Path) -> str:
    """Chuyển output path thành relative POSIX path đã kiểm tra phạm vi."""
    resolved_root = output_root.resolve()
    resolved_path = path.resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Output path nằm ngoài output_dir: {path}") from exc
    return relative_path.as_posix()


def _read_manifest_paths(manifest_path: Path) -> set[str]:
    """Đọc danh sách file từ manifest cũ; manifest hỏng không được dùng để xóa."""
    if not manifest_path.is_file():
        return set()
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.warn(
            f"Không đọc được output manifest cũ; bỏ qua cleanup: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return set()
    generated_files = data.get("generated_files") if isinstance(data, dict) else None
    if not isinstance(generated_files, list) or not all(
        isinstance(item, str) for item in generated_files
    ):
        warnings.warn(
            "Output manifest cũ không đúng schema; bỏ qua cleanup.",
            RuntimeWarning,
            stacklevel=2,
        )
        return set()
    return set(generated_files)


def _safe_manifest_target(output_root: Path, relative_path: str) -> Path | None:
    """Resolve một manifest path và bảo đảm nó vẫn nằm trong output root."""
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        return None
    candidate = output_root / path
    try:
        candidate.resolve().relative_to(output_root.resolve())
    except ValueError:
        return None
    return candidate


def main() -> None:
    """Thông báo hai entry point được hỗ trợ."""
    print(
        "Core arrays: import run_classification_evaluation and supply y_true, "
        "y_pred, y_proba, classes and class metadata."
    )
    print(
        "Machine-failure pipeline: python -m "
        "classification.machine_failure_pipeline"
    )


if __name__ == "__main__":
    main()
