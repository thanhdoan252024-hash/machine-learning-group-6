"""Audit the raw Student Exam Performance dataset for the primary case study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "regression" / "data" / "raw" / "student_exam_performance.csv"
OUTPUT = ROOT / "case_study" / "artifacts" / "regression_data_audit.json"
TARGET = "exam_score"

# Directly derived from the target according to the current project README.
TARGET_DERIVED = ["performance_grade", "pass_status", "performance_level"]
# Primary scenario: prediction before the exam is completed.
POST_EXAM_CANDIDATES = ["questions_attempted", "questions_correct"]


def main() -> None:
    df = pd.read_csv(DATA)
    numeric = df.select_dtypes(include=np.number)
    correlations = (
        numeric.corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .abs()
        .sort_values(ascending=False)
    )
    payload = {
        "schema_version": 1,
        "dataset": str(DATA.relative_to(ROOT)),
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "target": TARGET,
        "target_derived_columns_to_exclude": TARGET_DERIVED,
        "post_exam_columns_excluded_in_primary_scenario": POST_EXAM_CANDIDATES,
        "missing_values": {
            column: int(count)
            for column, count in df.isna().sum().items()
            if count > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_feature_abs_correlation_with_target_top10": {
            column: float(value) for column, value in correlations.head(10).items()
        },
        "questions_correct_abs_correlation": float(correlations.get("questions_correct", np.nan)),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
