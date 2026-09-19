"""Newton leaf value and split-gain primitives."""

import numpy as np


def soft_threshold(gradient_sum, reg_alpha):
    """L1 soft-thresholding used by LightGBM-style leaf optimization."""
    return np.sign(gradient_sum) * max(abs(float(gradient_sum)) - reg_alpha, 0.0)


def leaf_value(gradient_sum, hessian_sum, reg_alpha=0.0, reg_lambda=0.0):
    """Optimal second-order leaf value under L1/L2 regularization."""
    denominator = float(hessian_sum) + reg_lambda
    if denominator <= 0:
        return 0.0
    return -soft_threshold(gradient_sum, reg_alpha) / denominator


def regularized_score(gradient_sum, hessian_sum, reg_alpha=0.0, reg_lambda=0.0):
    """Leaf score used in split gain."""
    denominator = float(hessian_sum) + reg_lambda
    if denominator <= 0:
        return 0.0
    g = soft_threshold(gradient_sum, reg_alpha)
    return g * g / denominator


def split_gain(
    left_gradient,
    left_hessian,
    right_gradient,
    right_hessian,
    reg_alpha=0.0,
    reg_lambda=0.0,
):
    """Second-order gain from replacing one parent by two children."""
    parent_gradient = left_gradient + right_gradient
    parent_hessian = left_hessian + right_hessian
    return 0.5 * (
        regularized_score(left_gradient, left_hessian, reg_alpha, reg_lambda)
        + regularized_score(right_gradient, right_hessian, reg_alpha, reg_lambda)
        - regularized_score(parent_gradient, parent_hessian, reg_alpha, reg_lambda)
    )
