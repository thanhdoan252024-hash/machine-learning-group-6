"""Pipeline cho bai toan du doan nguy co mac benh tieu duong."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from classification.evaluation.run_machine_failure_evaluation import (
    evaluate_machine_failure_splits,
)
from classification.lightgbm_classification import LightGBMClassification


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = (
    REPO_ROOT / "classification" / "data"
    / "diabetes_binary_health_indicators_BRFSS2015.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "classification" / "evaluation" / "diabetes_outputs"
TARGET_COLUMN = "Diabetes_binary"


def load_diabetes_dataset(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Doc du lieu BRFSS va tra ve 21 feature va target 0/1."""
    resolved_path = Path(data_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Khong tim thay dataset: {resolved_path}")
    dataframe = pd.read_csv(resolved_path)
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Dataset phai co cot target {TARGET_COLUMN!r}.")
    features = dataframe.drop(columns=TARGET_COLUMN).copy()
    target = dataframe[TARGET_COLUMN].copy()
    if features.empty:
        raise ValueError("Dataset khong co feature nao.")
    if features.isna().any().any() or target.isna().any():
        raise ValueError("Dataset con gia tri thieu.")
    if set(target.unique()) != {0.0, 1.0}:
        raise ValueError("Diabetes_binary phai chua dung hai nhan 0 va 1.")
    return features, target.astype(int)


def split_diabetes_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chia train/test theo ty le lop de test phan anh phan bo goc."""
    return train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )


def build_diabetes_classifier(scale_pos_weight: float = 1.0) -> LightGBMClassification:
    """Cau hinh toi uu cho du lieu BRFSS: nhẹ regularization, dùng scale_pos_weight."""
    return LightGBMClassification(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=64,
        max_depth=7,
        min_child_samples=30,
        min_split_gain=0.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        feature_fraction=0.8,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )

def evaluate_diabetes_splits(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Danh gia train/test va xuat artifact rieng cho bai toan tieu duong."""
    return evaluate_machine_failure_splits(*args, **kwargs)


__all__ = [
    "DEFAULT_DATA_PATH", "DEFAULT_OUTPUT_DIR", "TARGET_COLUMN",
    "build_diabetes_classifier", "evaluate_diabetes_splits", "load_diabetes_dataset",
    "split_diabetes_dataset",
]
