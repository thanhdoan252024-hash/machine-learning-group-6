"""Leakage-safe regression pipeline for the LightGBM case study.

The historical notebook preprocesses the full dataset before train/test split.
This module defines the primary *pre-exam* scenario, splits raw rows first, and
fits every learned preprocessing statistic on the training split only.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from regression.lightgbm_regression import LightGBMRegression
from regression.metrics import mean_absolute_error, mean_squared_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "regression" / "data" / "raw" / "student_exam_performance.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "case_study" / "artifacts" / "regression"
TARGET_COLUMN = "exam_score"
IDENTIFIER_COLUMNS = ("student_id",)
TARGET_DERIVED_COLUMNS = ("performance_grade", "pass_status", "performance_level")
POST_EXAM_COLUMNS = ("questions_attempted", "questions_correct")
PRIMARY_EXCLUDED_COLUMNS = IDENTIFIER_COLUMNS + TARGET_DERIVED_COLUMNS + POST_EXAM_COLUMNS


@dataclass
class TrainOnlyPreprocessor:
    """Small pandas/numpy preprocessor with train-only learned state."""

    numeric_columns: list[str] | None = None
    categorical_columns: list[str] | None = None
    numeric_means: dict[str, float] | None = None
    categorical_modes: dict[str, object] | None = None
    category_maps: dict[str, dict[object, int]] | None = None
    feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame) -> "TrainOnlyPreprocessor":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X phải là pandas DataFrame.")
        self.numeric_columns = list(X.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = [
            column for column in X.columns if column not in self.numeric_columns
        ]
        self.numeric_means = {
            column: float(X[column].mean()) for column in self.numeric_columns
        }
        self.categorical_modes = {}
        self.category_maps = {}
        for column in self.categorical_columns:
            mode = X[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "__MISSING__"
            self.categorical_modes[column] = fill_value
            filled = X[column].fillna(fill_value)
            unique = list(dict.fromkeys(filled.tolist()))
            self.category_maps[column] = {
                value: index for index, value in enumerate(unique)
            }
        self.feature_names_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_names_ is None:
            raise RuntimeError("Cần gọi fit() trước transform().")
        if list(X.columns) != self.feature_names_:
            raise ValueError("Feature columns/order không khớp dữ liệu fit.")
        transformed = np.empty((len(X), len(self.feature_names_)), dtype=float)
        for j, column in enumerate(self.feature_names_):
            if column in self.numeric_columns:
                transformed[:, j] = pd.to_numeric(
                    X[column], errors="coerce"
                ).fillna(self.numeric_means[column]).to_numpy(dtype=float)
            else:
                filled = X[column].fillna(self.categorical_modes[column])
                mapping = self.category_maps[column]
                transformed[:, j] = np.array(
                    [mapping.get(value, -1) for value in filled], dtype=float
                )
        return transformed

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


def load_primary_regression_dataset(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the pre-exam primary scenario without target/post-exam leakage."""
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    required = {TARGET_COLUMN, *PRIMARY_EXCLUDED_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Dataset thiếu cột bắt buộc: " + ", ".join(missing))
    X = df.drop(columns=[TARGET_COLUMN, *PRIMARY_EXCLUDED_COLUMNS]).copy()
    y = df[TARGET_COLUMN].astype(float).copy()
    return X, y


def split_regression_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    """Split raw rows before any trainable preprocessing."""
    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("train/validation/test size phải có tổng bằng 1.")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(validation_size + test_size), random_state=random_state
    )
    test_fraction_of_temp = test_size / (validation_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_fraction_of_temp,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_regression_case_study(
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_params: dict[str, Any] | None = None,
    sample_size: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run one reproducible leakage-safe experiment and export artifacts.

    ``sample_size`` is intended only for smoke tests. Final case-study runs
    should leave it as ``None`` so all 100,000 rows are used.
    """
    X, y = load_primary_regression_dataset(data_path)
    if sample_size is not None:
        if sample_size < 100 or sample_size > len(X):
            raise ValueError("sample_size phải nằm trong [100, n_rows].")
        sampled = X.sample(n=sample_size, random_state=random_state).index
        X, y = X.loc[sampled].copy(), y.loc[sampled].copy()

    X_train, X_val, X_test, y_train, y_val, y_test = split_regression_dataset(
        X, y, random_state=random_state
    )
    preprocessor = TrainOnlyPreprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": -1,
        "max_bins": 32,
        "min_data_in_leaf": 20,
        "early_stopping_rounds": 15,
        "random_state": random_state,
    }
    if model_params:
        params.update(model_params)
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 15))
    model = LightGBMRegression(**params)
    model.fit(
        X_train_t,
        y_train.to_numpy(),
        eval_set=(X_val_t, y_val.to_numpy()),
        early_stopping_rounds=early_stopping_rounds,
    )

    predictions = {
        "train": model.predict(X_train_t),
        "validation": model.predict(X_val_t),
        "test": model.predict(X_test_t),
    }
    targets = {
        "train": y_train.to_numpy(),
        "validation": y_val.to_numpy(),
        "test": y_test.to_numpy(),
    }
    metrics = {
        split: regression_metrics(targets[split], predictions[split])
        for split in predictions
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "scenario": "pre_exam_prediction",
        "n_rows": int(len(X)),
        "split_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "excluded_columns": list(PRIMARY_EXCLUDED_COLUMNS),
        "n_features": int(X_train_t.shape[1]),
        "feature_names": preprocessor.feature_names_,
        "model_params": params,
        "early_stopping_rounds": early_stopping_rounds,
        "best_iteration": int(model.best_iteration_),
        "best_validation_rmse": float(model.best_score_),
        "metrics": metrics,
        "preprocessing_fit_split": "train",
        "model_selection_split": "validation",
        "primary_reporting_split": "test",
        "sampled_smoke_run": sample_size is not None,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    frames = []
    split_indices = {"train": y_train.index, "validation": y_val.index, "test": y_test.index}
    for split in ("train", "validation", "test"):
        frames.append(pd.DataFrame({
            "row_index": split_indices[split],
            "split": split,
            "y_true": targets[split],
            "y_pred": predictions[split],
            "residual": targets[split] - predictions[split],
        }))
    pd.concat(frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame({
        "feature": preprocessor.feature_names_,
        "gain_importance": model.get_feature_importance(),
    }).sort_values("gain_importance", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    history = pd.DataFrame({
        "iteration": np.arange(1, len(model.evals_result_["training"]["rmse"]) + 1),
        "train_rmse": model.evals_result_["training"]["rmse"],
        "validation_rmse": model.evals_result_["validation"]["rmse"],
    })
    history.to_csv(output_dir / "learning_history.csv", index=False)
    return payload


if __name__ == "__main__":
    result = run_regression_case_study()
    print(json.dumps(result, indent=2, ensure_ascii=False))
