"""Test tính split gain và regularized score."""

import numpy as np

from regression.lightgbm_regression import LightGBMRegression


def test_split_gain_zero_when_no_split():
    model = LightGBMRegression(reg_alpha=0.0, reg_lambda=1.0)
    # left = parent, right = 0 mẫu -> không có sự tách biệt
    gain = model._split_gain(10.0, 10.0, 10.0, 10.0)
    assert np.isclose(gain, 0.0)


def test_split_gain_positive_for_good_split():
    model = LightGBMRegression(reg_alpha=0.0, reg_lambda=0.0)
    # parent: g=0, h=10; left: g=-5, h=5; right: g=5, h=5
    # gain = 0.5*(25/5 + 25/5 - 0) = 5
    gain = model._split_gain(0.0, 10.0, -5.0, 5.0)
    assert gain > 0
    assert np.isclose(gain, 5.0)


def test_regularized_score_formula():
    model = LightGBMRegression(reg_alpha=0.0, reg_lambda=1.0)
    # score = g^2 / (h + lambda) = 4 / (2 + 1)
    assert np.isclose(model._regularized_score(2.0, 2.0), 4.0 / 3.0)


def test_split_gain_l2_reduces_gain():
    model_no_reg = LightGBMRegression(reg_alpha=0.0, reg_lambda=0.0)
    model_reg = LightGBMRegression(reg_alpha=0.0, reg_lambda=5.0)
    gain_no_reg = model_no_reg._split_gain(0.0, 10.0, -5.0, 5.0)
    gain_reg = model_reg._split_gain(0.0, 10.0, -5.0, 5.0)
    assert gain_reg < gain_no_reg
