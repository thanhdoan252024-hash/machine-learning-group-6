"""Test khả năng xử lý dữ liệu missing (NaN)."""

import numpy as np

from regression.lightgbm_regression import LightGBMRegression


def test_fit_predict_with_nan():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + rng.normal(scale=0.1, size=200)
    # Chèn NaN vào X (không ảnh hưởng y)
    X[0, 0] = np.nan
    X[5, 1] = np.nan
    X[10, 2] = np.nan
    X[15, 0] = np.nan
    model = LightGBMRegression(n_estimators=10, num_leaves=8, max_bins=16)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (200,)
    assert np.all(np.isfinite(preds))


def test_missing_value_gets_max_bin():
    X = np.array([[1.0], [np.nan], [3.0], [4.0]])
    model = LightGBMRegression(max_bins=4)
    model._prepare_bins(X)
    # Giá trị missing được gán bin = max_bins
    assert model.feature_bins[1, 0] == model.max_bins
    assert model.feature_bins[0, 0] != model.max_bins


def test_predict_with_nan_uses_missing_branch():
    X_train = np.random.default_rng(0).normal(size=(100, 2))
    y_train = 2.0 * X_train[:, 0] + np.random.default_rng(1).normal(scale=0.1, size=100)
    model = LightGBMRegression(n_estimators=8, num_leaves=8, max_bins=16)
    model.fit(X_train, y_train)
    X_test = np.array([[np.nan, 0.5], [1.0, np.nan]])
    preds = model.predict(X_test)
    assert preds.shape == (2,)
    assert np.all(np.isfinite(preds))


def test_all_missing_column_handled():
    X = np.full((20, 2), np.nan)
    y = np.random.default_rng(0).normal(size=20)
    model = LightGBMRegression(n_estimators=5, num_leaves=4, max_bins=8)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))
