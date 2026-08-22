import numpy as np


def mean_squared_error(y_true, y_pred):
    """Tính Mean Squared Error (MSE)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def r_squared(y_true, y_pred):
    """Tính R-squared / Coefficient of Determination (R2 Score)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1 - (ss_res / ss_tot))

