"""Test tính giá trị leaf (leaf value) và regularization."""

import numpy as np

from regression.lightgbm_regression import LightGBMRegression


def test_leaf_value_negative_gradient_over_hessian():
    model = LightGBMRegression(reg_alpha=0.0, reg_lambda=0.0)
    # leaf_value = -g / h = -4/2 = -2
    assert np.isclose(model._leaf_value(4.0, 2.0), -2.0)


def test_leaf_value_l1_shrinkage_to_zero():
    model = LightGBMRegression(reg_alpha=2.0, reg_lambda=0.0)
    # |g| = 1 < alpha=2 -> soft-threshold về 0
    assert np.isclose(model._leaf_value(1.0, 1.0), 0.0)


def test_leaf_value_l1_after_threshold():
    model = LightGBMRegression(reg_alpha=2.0, reg_lambda=0.0)
    # |g|=5, alpha=2 -> 3 -> -3/1 = -3
    assert np.isclose(model._leaf_value(5.0, 1.0), -3.0)


def test_leaf_value_l2_regularization():
    model = LightGBMRegression(reg_alpha=0.0, reg_lambda=3.0)
    # -g/(h+lambda) = -4/(1+3) = -1
    assert np.isclose(model._leaf_value(4.0, 1.0), -1.0)


def test_leaf_value_positive_gradient_positive_value():
    model = LightGBMRegression(reg_alpha=0.0, reg_lambda=0.0)
    # g = -3 -> value = 3
    assert np.isclose(model._leaf_value(-3.0, 1.0), 3.0)
