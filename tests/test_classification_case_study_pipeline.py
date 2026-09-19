import numpy as np
import pandas as pd

from classification.case_study_pipeline import (
    binary_metrics,
    split_classification_dataset,
    tune_threshold,
)


def test_classification_split_is_70_15_15_stratified_and_disjoint():
    X = pd.DataFrame({"x": np.arange(1000)})
    y = pd.Series([0] * 900 + [1] * 100)
    Xtr, Xv, Xte, ytr, yv, yte = split_classification_dataset(X, y)
    assert (len(Xtr), len(Xv), len(Xte)) == (700, 150, 150)
    assert set(Xtr.index).isdisjoint(Xv.index)
    assert set(Xtr.index).isdisjoint(Xte.index)
    assert set(Xv.index).isdisjoint(Xte.index)
    assert np.isclose(ytr.mean(), 0.1)
    assert np.isclose(yv.mean(), 0.1)
    assert np.isclose(yte.mean(), 0.1)


def test_binary_metrics_include_imbalance_sensitive_metrics():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.05, 0.2, 0.6, 0.4, 0.9])
    metrics = binary_metrics(y, p, 0.5)
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 2
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc_average_precision"] <= 1
    assert 0 <= metrics["balanced_accuracy"] <= 1


def test_threshold_tuning_uses_validation_f1():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.45, 0.9])
    threshold, frame = tune_threshold(y, p, thresholds=[0.3, 0.5])
    assert threshold == 0.3
    assert set(frame["threshold"]) == {0.3, 0.5}
