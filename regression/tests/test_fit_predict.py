"""Test chức năng fit và predict của LightGBMRegression."""

import numpy as np
import pytest

from regression.lightgbm_regression import LightGBMRegression
from regression.tests._helpers import make_regression_data


def test_fit_requires_2d_X():
    model = LightGBMRegression(n_estimators=5)
    with pytest.raises(ValueError):
        model.fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))


def test_fit_requires_matching_rows():
    model = LightGBMRegression(n_estimators=5)
    X = np.zeros((5, 2))
    y = np.zeros(4)
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_predict_requires_fit():
    model = LightGBMRegression()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((3, 2)))


def test_predict_shape():
    X, y = make_regression_data()
    model = LightGBMRegression(n_estimators=5, num_leaves=8)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)


def test_predict_requires_same_features():
    X, y = make_regression_data(n_features=3)
    model = LightGBMRegression(n_estimators=5)
    model.fit(X, y)
    with pytest.raises(ValueError):
        model.predict(np.zeros((3, 5)))  # 5 cột != 3 cột lúc huấn luyện


def test_fit_reduces_error():
    X, y = make_regression_data(n=300, noise=0.2)
    model = LightGBMRegression(n_estimators=30, num_leaves=16, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    base_mae = np.mean(np.abs(y - y.mean()))
    model_mae = np.mean(np.abs(y - preds))
    assert model_mae < base_mae


def test_predict_is_finite():
    X, y = make_regression_data()
    model = LightGBMRegression(n_estimators=5, num_leaves=8)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))


def test_base_prediction_is_mean():
    X, y = make_regression_data()
    model = LightGBMRegression(n_estimators=1, num_leaves=2)
    model.fit(X, y)
    assert np.isclose(model.base_prediction, y.mean())


def test_feature_importance_shape():
    X, y = make_regression_data(n_features=4)
    model = LightGBMRegression(n_estimators=5, num_leaves=8)
    model.fit(X, y)
    assert model.get_feature_importance().shape == (4,)
