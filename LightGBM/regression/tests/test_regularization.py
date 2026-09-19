"""Test regularization lambda_l1 (reg_alpha) và lambda_l2 (reg_lambda)."""

import numpy as np

from regression.lightgbm_regression import LightGBMRegression


def test_l1_lambda_reduces_leaf_magnitude():
    model_no_reg = LightGBMRegression(reg_alpha=0.0, reg_lambda=0.0)
    model_l1 = LightGBMRegression(reg_alpha=5.0, reg_lambda=0.0)
    value_no_reg = model_no_reg._leaf_value(3.0, 1.0)  # -3
    value_l1 = model_l1._leaf_value(3.0, 1.0)  # soft-threshold: 3-5 -> 0
    assert abs(value_l1) < abs(value_no_reg)
    assert np.isclose(value_l1, 0.0)


def test_l1_lambda_shrinks_small_gradients_to_zero():
    model = LightGBMRegression(reg_alpha=1.5, reg_lambda=0.0)
    assert np.isclose(model._leaf_value(1.0, 1.0), 0.0)


def test_l2_lambda_reduces_leaf_magnitude():
    model_no_reg = LightGBMRegression(reg_alpha=0.0, reg_lambda=0.0)
    model_l2 = LightGBMRegression(reg_alpha=0.0, reg_lambda=10.0)
    value_no_reg = model_no_reg._leaf_value(4.0, 1.0)  # -4
    value_l2 = model_l2._leaf_value(4.0, 1.0)  # -4/11
    assert abs(value_l2) < abs(value_no_reg)
    assert np.isclose(value_l2, -4.0 / 11.0)


def test_l2_lambda_reduces_split_gain():
    model_no_reg = LightGBMRegression(reg_alpha=0.0, reg_lambda=0.0)
    model_l2 = LightGBMRegression(reg_alpha=0.0, reg_lambda=10.0)
    gain_no_reg = model_no_reg._split_gain(0.0, 10.0, -5.0, 5.0)
    gain_l2 = model_l2._split_gain(0.0, 10.0, -5.0, 5.0)
    assert gain_l2 < gain_no_reg


def test_negative_lambda_raises():
    model = LightGBMRegression(reg_alpha=-1.0, reg_lambda=0.0)
    X = np.random.default_rng(0).normal(size=(10, 2))
    y = np.random.default_rng(1).normal(size=10)
    try:
        model.fit(X, y)
        raise AssertionError("Phải raise ValueError cho reg_alpha âm")
    except ValueError:
        pass
