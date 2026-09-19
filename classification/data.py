"""Dataset definitions and loading helpers for machine-failure experiments.

This module is intentionally protocol-neutral.  The final 70/15/15 case-study
pipeline and the archived Phase-1 80/20 baseline both import the same loader so
that data preparation is not coupled to either experimental protocol.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "classification" / "data" / "raw" / "machine_fail.csv"
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
    """Read the six non-leaking features and the binary machine-failure target."""
    resolved_path = Path(data_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset: {resolved_path}")

    dataframe = pd.read_csv(resolved_path)
    required_columns = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        raise ValueError("Dataset thiếu các cột bắt buộc: " + ", ".join(missing_columns))

    features = dataframe.loc[:, FEATURE_COLUMNS].copy()
    encoded_type = features["Type"].map(TYPE_ENCODING)
    if encoded_type.isna().any():
        unknown_values = sorted(
            str(value) for value in features.loc[encoded_type.isna(), "Type"].unique()
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
