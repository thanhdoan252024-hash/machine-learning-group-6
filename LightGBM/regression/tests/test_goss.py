"""Test cơ chế GOSS (Gradient-based One-Side Sampling)."""

import numpy as np

from regression.lightgbm_regression import LightGBMRegression


def test_goss_selects_top_gradient_rows():
    model = LightGBMRegression(top_rate=0.2, other_rate=0.1, random_state=0)
    gradients = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    selected, _ = model._goss_sample(gradients, 0)
    # top 20% = 2 dòng có |gradient| lớn nhất: index 8 và 9
    assert 8 in selected and 9 in selected


def test_goss_sample_size():
    model = LightGBMRegression(top_rate=0.2, other_rate=0.1, random_state=0)
    gradients = np.arange(1.0, 21.0)  # 20 dòng
    selected, _ = model._goss_sample(gradients, 0)
    # top 20% = 4, other 10% = 2 -> tổng 6
    assert len(selected) == 6


def test_goss_other_rows_are_weighted():
    model = LightGBMRegression(top_rate=0.2, other_rate=0.1, random_state=0)
    gradients = np.arange(1.0, 21.0)
    selected, weights = model._goss_sample(gradients, 0)
    other_weight = (1.0 - model.top_rate) / model.other_rate  # 8.0
    n_other = np.count_nonzero(np.isclose(weights, other_weight))
    assert n_other == 2
    # Các dòng top có weight = 1
    top_weight_count = np.count_nonzero(np.isclose(weights[selected], 1.0))
    assert top_weight_count == 4


def test_goss_weights_shape():
    model = LightGBMRegression(top_rate=0.2, other_rate=0.1, random_state=0)
    gradients = np.arange(1.0, 21.0)
    selected, weights = model._goss_sample(gradients, 0)
    assert weights.shape == gradients.shape


def test_goss_reproducible_with_seed():
    model = LightGBMRegression(top_rate=0.2, other_rate=0.1, random_state=7)
    gradients = np.arange(1.0, 31.0)
    s1, _ = model._goss_sample(gradients, 0)
    s2, _ = model._goss_sample(gradients, 0)
    np.testing.assert_array_equal(np.sort(s1), np.sort(s2))
