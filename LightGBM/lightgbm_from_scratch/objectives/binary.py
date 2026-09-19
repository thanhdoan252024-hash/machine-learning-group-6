"""Binary classification objective."""

import numpy as np


def sigmoid(raw_score):
    """Numerically stable logistic transform."""
    raw_score = np.asarray(raw_score, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(raw_score, -35.0, 35.0)))


def binary_logloss_gradient_hessian(y_true, raw_score, min_hessian=1e-12):
    """Gradient/Hessian of binary log-loss with respect to raw score."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    raw_score = np.asarray(raw_score, dtype=float).reshape(-1)
    if y_true.shape != raw_score.shape:
        raise ValueError("y_true và raw_score phải có cùng kích thước.")
    probability = sigmoid(raw_score)
    gradient = probability - y_true
    hessian = np.maximum(probability * (1.0 - probability), min_hessian)
    return gradient, hessian
