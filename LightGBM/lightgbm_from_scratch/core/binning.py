"""Quantile histogram binning shared by both tasks."""

import numpy as np


def create_quantile_bins(X, max_bins, *, quantile_style="boundaries"):
    """Learn sorted quantile cut points from finite training values only.

    ``quantile_style='boundaries'`` creates ``max_bins-1`` candidate cuts,
    matching the historical regression estimator. ``'centers'`` preserves the
    classification project's original ``linspace(0, 1, max_bins)[1:-1]`` rule.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X phải là ma trận hai chiều.")
    if max_bins < 2:
        raise ValueError("max_bins phải >= 2.")

    if quantile_style == "boundaries":
        quantiles = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
    elif quantile_style == "centers":
        quantiles = np.linspace(0.0, 1.0, max_bins)[1:-1]
    else:
        raise ValueError("quantile_style không hợp lệ.")

    cuts = []
    for column in X.T:
        valid = column[np.isfinite(column)]
        if valid.size == 0 or (quantile_style == "centers" and (valid.size <= 1 or np.ptp(valid) == 0)):
            cuts.append(np.array([], dtype=float))
        else:
            cuts.append(np.unique(np.quantile(valid, quantiles)))
    return cuts


def bin_matrix(X, cuts, *, missing_bin="zero", max_bins=None):
    """Transform data using training cut points without refitting.

    ``missing_bin='zero'``: missing=0 and valid bins start at 1.
    ``missing_bin='last'``: missing=max_bins and valid bins start at 0.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(cuts):
        raise ValueError("X không phù hợp với số feature đã fit bins.")

    if missing_bin == "zero":
        result = np.zeros(X.shape, dtype=np.int32)
        for j, thresholds in enumerate(cuts):
            valid = np.isfinite(X[:, j])
            result[valid, j] = np.searchsorted(
                thresholds, X[valid, j], side="right"
            ) + 1
        return result

    if missing_bin == "last":
        if max_bins is None:
            raise ValueError("Cần max_bins khi missing_bin='last'.")
        result = np.full(X.shape, int(max_bins), dtype=np.int32)
        for j, thresholds in enumerate(cuts):
            valid = np.isfinite(X[:, j])
            result[valid, j] = np.searchsorted(
                thresholds, X[valid, j], side="right"
            )
        return result

    raise ValueError("missing_bin phải là 'zero' hoặc 'last'.")
