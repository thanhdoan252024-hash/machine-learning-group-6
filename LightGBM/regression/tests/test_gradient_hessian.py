"""Test gradient và hessian của thuật toán regression."""

import numpy as np

from regression.lightgbm_regression import LightGBMRegression


def test_gradient_is_prediction_minus_true():
    y_true = np.array([1.0, 2.0, 3.0])
    preds = np.array([1.5, 1.0, 3.0])
    gradients, _ = LightGBMRegression._compute_gradient_hessian(y_true, preds)
    np.testing.assert_allclose(gradients, preds - y_true)


def test_hessian_is_ones():
    y_true = np.array([0.0, 5.0, -3.0])
    preds = np.array([1.0, 1.0, 1.0])
    _, hessians = LightGBMRegression._compute_gradient_hessian(y_true, preds)
    np.testing.assert_allclose(hessians, np.ones(3))


def test_gradient_hessian_shape():
    y_true = np.zeros(10)
    preds = np.zeros(10)
    gradients, hessians = LightGBMRegression._compute_gradient_hessian(y_true, preds)
    assert gradients.shape == (10,)
    assert hessians.shape == (10,)


def test_gradient_zero_when_perfect_prediction():
    y_true = np.array([2.0, 4.0, 6.0])
    preds = np.array([2.0, 4.0, 6.0])
    gradients, _ = LightGBMRegression._compute_gradient_hessian(y_true, preds)
    np.testing.assert_allclose(gradients, np.zeros(3))

