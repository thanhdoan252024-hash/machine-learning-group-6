"""Regression objectives."""

import numpy as np


def squared_error_gradient_hessian(y_true, predictions):
    """Gradient/Hessian for ``0.5 * (prediction - target)^2``."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if y_true.shape != predictions.shape:
        raise ValueError("y_true và predictions phải có cùng kích thước.")
    gradients = predictions - y_true
    hessians = np.ones_like(gradients)
    return gradients, hessians
