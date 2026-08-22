"""Pipeline tái lập cho bài toán dự đoán hỏng máy.

``scikit-learn`` chỉ được dùng để chia train/test có stratification. Model,
metric, ROC-AUC, báo cáo và biểu đồ đều dùng implementation của repository.
"""

import json
from pathlib import Path
from typing import Any, Mapping
import uuid
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from classification.evaluation.adapter import (
    evaluate_classification_outputs,
)
from classification.lightgbm_classification import LightGBMClassification


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "classification" / "data" / "raw" / "machine_fail.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "classification" / "evaluation" / "outputs"

TARGET_COLUMN = "Machine failure"
FEATURE_COLUMNS = (
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
TYPE_ENCODING = {"L": 0, "M": 1, "H": 2}


def load_machine_failure_dataset(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Đọc dataset và tạo đúng sáu feature không gây target leakage."""

    resolved_path = Path(data_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset: {resolved_path}")

    dataframe = pd.read_csv(resolved_path)
    required_columns = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(
            "Dataset thiếu các cột bắt buộc: " + ", ".join(missing_columns)
        )

    features = dataframe.loc[:, FEATURE_COLUMNS].copy()
    encoded_type = features["Type"].map(TYPE_ENCODING)
    if encoded_type.isna().any():
        unknown_values = sorted(
            str(value)
            for value in features.loc[encoded_type.isna(), "Type"].unique()
        )
        raise ValueError(
            "Cột Type chứa giá trị chưa được ánh xạ: " + ", ".join(unknown_values)
        )
    features["Type"] = encoded_type.astype(int)

    target = dataframe[TARGET_COLUMN].copy()
    if features.isna().any().any() or target.isna().any():
        raise ValueError("Dataset còn giá trị thiếu trong feature hoặc target.")
    if set(target.unique()) != {0, 1}:
        raise ValueError("Target Machine failure phải chứa đúng hai nhãn 0 và 1.")

    return features, target


def split_machine_failure_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chia dữ liệu có stratification; đây là phạm vi duy nhất dùng sklearn."""

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def build_machine_failure_classifier() -> LightGBMClassification:
    """Khởi tạo đúng cấu hình model trong notebook với seed tái lập."""

    return LightGBMClassification(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=15,
        max_depth=5,
        random_state=42,
    )


def evaluate_machine_failure_splits(
    model: LightGBMClassification,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    save_dpi: int = 300,
) -> dict[str, Any]:
    """Đánh giá cùng một model đã fit trên train và test.

    Mỗi split có namespace output riêng để các tên artifact cố định của core
    runner không ghi đè lẫn nhau. Kết quả test vẫn được đưa lên top-level để
    tương thích với API trước đây; kết quả đầy đủ nằm trong ``split_results``.
    Manifest gốc chỉ được cập nhật sau khi cả hai evaluation thành công.
    """

    output_root = Path(output_dir)
    split_inputs = {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
    }
    split_results: dict[str, dict[str, Any]] = {}

    for split_name in split_inputs:
        _validate_split_output_root(output_root, split_name)

    for split_name, (features, target) in split_inputs.items():
        y_pred = model.predict(features)
        y_proba = model.predict_proba(features)
        split_results[split_name] = evaluate_classification_outputs(
            model=model,
            y_true=target.to_numpy(),
            y_pred=y_pred,
            y_proba=y_proba,
            output_dir=output_root / split_name,
            save_dpi=save_dpi,
        )

    split_sizes = {
        "train": len(X_train),
        "test": len(X_test),
    }
    split_class_counts = {
        "train": _class_counts(y_train),
        "test": _class_counts(y_test),
    }
    pipeline_manifest_path = _write_pipeline_manifest(
        output_root,
        split_results,
        split_sizes,
        split_class_counts,
        float(model.threshold),
    )

    # Test là primary reporting split để giữ tương thích với result cũ.
    result = dict(split_results["test"])
    result.update(
        {
            "manifest_path": pipeline_manifest_path,
            "pipeline_manifest_path": pipeline_manifest_path,
            "primary_reporting_split": "test",
            "split_results": split_results,
        }
    )
    return result


def run_machine_failure_pipeline(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    save_dpi: int = 300,
) -> dict[str, Any]:
    """Fit model một lần rồi xuất đánh giá độc lập cho train và test."""

    features, target = load_machine_failure_dataset(data_path)
    X_train, X_test, y_train, y_test = split_machine_failure_dataset(
        features,
        target,
    )

    model = build_machine_failure_classifier()
    model.fit(X_train, y_train)
    result = evaluate_machine_failure_splits(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        output_dir=output_dir,
        save_dpi=save_dpi,
    )
    result["pipeline_metadata"] = {
        "data_path": Path(data_path).resolve(),
        "output_dir": Path(output_dir).resolve(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_names": list(FEATURE_COLUMNS),
        "random_state": 42,
        "test_size": 0.2,
        "model_threshold": model.threshold,
        "model_fit_split": "train",
        "primary_reporting_split": "test",
    }
    return result


def _write_pipeline_manifest(
    output_root: Path,
    split_results: Mapping[str, Mapping[str, Any]],
    split_sizes: Mapping[str, int],
    split_class_counts: Mapping[str, Mapping[str, int]],
    model_threshold: float,
) -> Path:
    """Ghi manifest schema v2 và migrate artifact root schema v1 an toàn.

    Child manifests là nguồn ownership cho artifact từng split. Cleanup legacy
    không dùng glob, chỉ xét ``generated_files`` hợp lệ của manifest schema v1,
    và diễn ra sau khi cả train lẫn test đã đánh giá thành công.
    """

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "evaluation_manifest.json"
    legacy_paths = _read_legacy_generated_files(manifest_path)

    split_entries: dict[str, dict[str, Any]] = {}
    owned_files: set[str] = set()
    for split_name in ("train", "test"):
        manifest_relative, artifact_paths = _read_child_ownership(
            output_root,
            split_name,
            split_results[split_name],
        )
        split_entries[split_name] = {
            "n_samples": int(split_sizes[split_name]),
            "class_counts": dict(split_class_counts[split_name]),
            "manifest_path": manifest_relative,
            "generated_files": sorted(artifact_paths),
        }
        owned_files.add(manifest_relative)
        owned_files.update(artifact_paths)

    manifest = {
        "schema_version": 2,
        "task_type": "binary",
        "model_fit_split": "train",
        "primary_reporting_split": "test",
        "model_threshold": model_threshold,
        "splits": split_entries,
        "generated_files": sorted(owned_files),
    }
    temporary_path = manifest_path.with_name(
        f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        temporary_path.replace(manifest_path)
        try:
            _cleanup_legacy_generated_files(
                output_root,
                manifest_path,
                legacy_paths,
                owned_files,
            )
        except OSError as exc:
            warnings.warn(
                f"Root manifest v2 đã ghi nhưng cleanup legacy thất bại: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    finally:
        if temporary_path.is_file():
            temporary_path.unlink()

    return manifest_path


def _class_counts(target: pd.Series) -> dict[str, int]:
    """Đếm nhãn với key chuỗi để JSON ổn định và không phụ thuộc NumPy scalar."""

    counts = target.value_counts()
    return {
        str(label): int(counts[label])
        for label in sorted(counts.index, key=lambda value: str(value))
    }


def _read_child_ownership(
    output_root: Path,
    split_name: str,
    split_result: Mapping[str, Any],
) -> tuple[str, set[str]]:
    """Đọc và validate ownership từ manifest vừa sinh của một split."""

    raw_manifest_path = split_result.get("manifest_path")
    if not isinstance(raw_manifest_path, (str, Path)):
        raise ValueError(f"Split {split_name!r} không trả về manifest_path hợp lệ.")

    manifest_path = Path(raw_manifest_path)
    split_root = output_root / split_name
    manifest_relative = _relative_output_path(manifest_path, output_root)
    if manifest_path.resolve().parent != split_root.resolve():
        raise ValueError(
            f"Manifest của split {split_name!r} phải nằm trực tiếp trong "
            f"{split_root}."
        )

    try:
        child_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Không đọc được manifest của split {split_name!r}: {exc}"
        ) from exc

    if not isinstance(child_manifest, dict):
        raise ValueError(
            f"Manifest của split {split_name!r} không đúng schema v1."
        )
    generated_files = child_manifest.get("generated_files")
    if (
        child_manifest.get("schema_version") != 1
        or not isinstance(generated_files, list)
        or not all(isinstance(item, str) for item in generated_files)
    ):
        raise ValueError(
            f"Manifest của split {split_name!r} không đúng schema v1."
        )

    artifact_paths: set[str] = set()
    for relative_path in generated_files:
        target = _safe_manifest_target(split_root, relative_path)
        if target is None:
            raise ValueError(
                f"Manifest của split {split_name!r} chứa path không an toàn: "
                f"{relative_path!r}."
            )
        if not target.is_file():
            raise ValueError(
                f"Artifact của split {split_name!r} không tồn tại hoặc không "
                f"phải file: {relative_path!r}."
            )
        if target.stat().st_size == 0:
            raise ValueError(
                f"Artifact của split {split_name!r} là file rỗng: "
                f"{relative_path!r}."
            )
        artifact_paths.add(_relative_output_path(target, output_root))
    return manifest_relative, artifact_paths


def _read_legacy_generated_files(manifest_path: Path) -> set[str]:
    """Đọc ownership schema v1; manifest khác schema không được dùng để xóa."""

    if not manifest_path.is_file():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        warnings.warn(
            f"Không đọc được root output manifest cũ; bỏ qua cleanup: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return set()

    if isinstance(manifest, dict) and manifest.get("schema_version") == 2:
        return set()
    generated_files = (
        manifest.get("generated_files") if isinstance(manifest, dict) else None
    )
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or not isinstance(generated_files, list)
        or not all(isinstance(item, str) for item in generated_files)
    ):
        warnings.warn(
            "Root output manifest cũ không đúng schema v1; bỏ qua cleanup.",
            RuntimeWarning,
            stacklevel=2,
        )
        return set()
    return set(generated_files)


def _cleanup_legacy_generated_files(
    output_root: Path,
    manifest_path: Path,
    legacy_paths: set[str],
    owned_files: set[str],
) -> None:
    """Xóa đúng file legacy được manifest sở hữu, không xóa directory."""

    protected_paths = {manifest_path.resolve()}
    for relative_path in owned_files:
        target = _safe_manifest_target(output_root, relative_path)
        if target is not None:
            protected_paths.add(target.resolve())

    for relative_path in sorted(legacy_paths):
        target = _safe_manifest_target(output_root, relative_path)
        if target is None:
            warnings.warn(
                f"Bỏ qua path legacy không an toàn: {relative_path!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if target.resolve() in protected_paths:
            continue
        if target.is_file() or target.is_symlink():
            try:
                target.unlink()
            except OSError as exc:
                warnings.warn(
                    f"Không xóa được legacy output {target}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif target.exists():
            warnings.warn(
                f"Không xóa legacy output vì path không phải file: {target}",
                RuntimeWarning,
                stacklevel=2,
            )


def _safe_manifest_target(output_root: Path, relative_path: str) -> Path | None:
    """Resolve manifest path và bảo đảm target là con của output root."""

    path = Path(relative_path)
    if not path.parts or path.drive or path.is_absolute() or ".." in path.parts:
        return None
    candidate = output_root / path
    try:
        candidate.resolve().relative_to(output_root.resolve())
    except ValueError:
        return None
    if candidate.resolve() == output_root.resolve():
        return None
    return candidate


def _validate_split_output_root(output_root: Path, split_name: str) -> None:
    """Chặn split namespace resolve ra ngoài output root qua symlink/path lạ."""

    split_root = output_root / split_name
    try:
        split_root.resolve().relative_to(output_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Output root của split {split_name!r} nằm ngoài output_dir."
        ) from exc


def _relative_output_path(path: Path, output_root: Path) -> str:
    """Trả về POSIX relative path và từ chối path ngoài output root."""

    try:
        relative_path = path.resolve().relative_to(output_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Output path nằm ngoài output_dir: {path}") from exc
    return relative_path.as_posix()


def _print_split_metrics(
    split_name: str,
    split_result: Mapping[str, Any],
    n_samples: int,
) -> None:
    """In metric ASCII ổn định cho một split."""

    metrics = split_result["metrics"]
    print(f"{split_name.title()}: {n_samples} samples")
    print(f"  Accuracy:  {metrics['accuracy']:.6f}")
    print(f"  Precision: {metrics['precision']:.6f}")
    print(f"  Recall:    {metrics['recall']:.6f}")
    print(f"  F1-score:  {metrics['f1_score']:.6f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.6f}")


def main() -> None:
    """CLI entry point dùng để tái tạo các CSV/PNG đã commit."""

    result = run_machine_failure_pipeline()
    metadata = result["pipeline_metadata"]
    # Dùng ASCII để CLI chạy ổn định cả trên Windows console dùng CP1252.
    _print_split_metrics(
        "train",
        result["split_results"]["train"],
        metadata["n_train"],
    )
    _print_split_metrics(
        "test",
        result["split_results"]["test"],
        metadata["n_test"],
    )
    print(f"Artifacts: {metadata['output_dir']}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "build_machine_failure_classifier",
    "evaluate_machine_failure_splits",
    "load_machine_failure_dataset",
    "run_machine_failure_pipeline",
    "split_machine_failure_dataset",
]
