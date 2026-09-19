import numpy as np
import pandas as pd

from regression.case_study_pipeline import (
    POST_EXAM_COLUMNS,
    PRIMARY_EXCLUDED_COLUMNS,
    TrainOnlyPreprocessor,
    load_primary_regression_dataset,
    split_regression_dataset,
)


def test_primary_scenario_excludes_post_exam_and_target_derived_columns():
    X, y = load_primary_regression_dataset()
    assert len(X) == len(y) == 100_000
    for column in PRIMARY_EXCLUDED_COLUMNS:
        assert column not in X.columns
    assert "exam_score" not in X.columns
    assert all(column not in X.columns for column in POST_EXAM_COLUMNS)


def test_split_is_70_15_15_and_disjoint():
    X = pd.DataFrame({"x": np.arange(1000)})
    y = pd.Series(np.arange(1000))
    Xtr, Xv, Xte, ytr, yv, yte = split_regression_dataset(X, y)
    assert (len(Xtr), len(Xv), len(Xte)) == (700, 150, 150)
    assert set(Xtr.index).isdisjoint(Xv.index)
    assert set(Xtr.index).isdisjoint(Xte.index)
    assert set(Xv.index).isdisjoint(Xte.index)
    assert ytr.index.equals(Xtr.index)
    assert yv.index.equals(Xv.index)
    assert yte.index.equals(Xte.index)


def test_preprocessor_uses_training_mean_and_unknown_category_code():
    train = pd.DataFrame({
        "numeric": [1.0, 3.0, np.nan],
        "category": ["A", "A", "B"],
    })
    test = pd.DataFrame({
        "numeric": [1000.0, np.nan],
        "category": ["UNSEEN", "A"],
    })
    pre = TrainOnlyPreprocessor().fit(train)
    transformed = pre.transform(test)
    assert np.isclose(pre.numeric_means["numeric"], 2.0)
    assert np.isclose(transformed[1, 0], 2.0)
    assert transformed[0, 1] == -1.0
