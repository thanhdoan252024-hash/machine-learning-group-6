"""Pipeline tái lập cho bài toán dự đoán hỏng máy.

``scikit-learn`` chỉ được dùng để chia train/test có stratification. Model,
metric, ROC-AUC, báo cáo và biểu đồ đều dùng implementation của repository.
"""

from pathlib import Path
from typing import Any

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


def run_machine_failure_pipeline(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    save_dpi: int = 300,
) -> dict[str, Any]:
    """Train model, dự đoán test split và xuất trọn bộ kết quả đánh giá."""

    features, target = load_machine_failure_dataset(data_path)
    X_train, X_test, y_train, y_test = split_machine_failure_dataset(
        features,
        target,
    )

    model = build_machine_failure_classifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    result = evaluate_classification_outputs(
        model=model,
        y_true=y_test.to_numpy(),
        y_pred=y_pred,
        y_proba=y_proba,
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
    }
    return result


def main() -> None:
    """CLI entry point dùng để tái tạo các CSV/PNG đã commit."""

    result = run_machine_failure_pipeline()
    metrics = result["metrics"]
    metadata = result["pipeline_metadata"]
    # Dùng ASCII để CLI chạy ổn định cả trên Windows console dùng CP1252.
    print(f"Evaluated {metadata['n_test']} test samples.")
    print(f"Accuracy:  {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall:    {metrics['recall']:.6f}")
    print(f"F1-score:  {metrics['f1_score']:.6f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.6f}")
    print(f"Artifacts: {metadata['output_dir']}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "build_machine_failure_classifier",
    "load_machine_failure_dataset",
    "run_machine_failure_pipeline",
    "split_machine_failure_dataset",
]
