"""Shared histogram split search."""

from __future__ import annotations

import numpy as np

from .regularization import split_gain


def find_best_histogram_split(
    histograms,
    features,
    *,
    missing_bin,
    reg_alpha=0.0,
    reg_lambda=1.0,
    min_gain=0.0,
    min_child_samples=1,
    min_child_weight=0.0,
):
    """Find the best ``bin <= threshold`` split across feature histograms.

    Histogram rows are ``gradient, hessian`` and optionally ``count``. If
    counts are absent, the sample-count constraint is deferred to the tree
    builder (the historical regression behavior).
    """
    best = None
    best_gain = float(min_gain)

    for feature in map(int, features):
        histogram = np.asarray(histograms[feature], dtype=float)
        if histogram.ndim != 2 or histogram.shape[0] not in (2, 3):
            raise ValueError("Histogram phải có 2 hoặc 3 hàng.")
        n_bins = histogram.shape[1]
        feature_missing_bin = int(missing_bin(feature) if callable(missing_bin) else missing_bin)
        if feature_missing_bin < 0 or feature_missing_bin >= n_bins:
            raise ValueError("missing_bin nằm ngoài phạm vi histogram.")

        valid_bins = [index for index in range(n_bins) if index != feature_missing_bin]
        if len(valid_bins) < 2:
            continue
        missing = histogram[:, feature_missing_bin]
        nonmissing_total = histogram[:, valid_bins].sum(axis=1)
        cumulative = np.zeros(histogram.shape[0], dtype=float)

        # The final valid bin cannot define a non-empty right side.
        for threshold_bin in valid_bins[:-1]:
            cumulative += histogram[:, threshold_bin]
            for missing_left in (True, False):
                left = cumulative.copy()
                if missing_left:
                    left += missing
                total = nonmissing_total + missing
                right = total - left

                if histogram.shape[0] == 3:
                    if left[2] < min_child_samples or right[2] < min_child_samples:
                        continue
                if left[1] < min_child_weight or right[1] < min_child_weight:
                    continue
                if left[1] <= 0 or right[1] <= 0:
                    continue

                gain = split_gain(
                    left[0], left[1], right[0], right[1], reg_alpha, reg_lambda
                )
                if gain > best_gain:
                    best_gain = float(gain)
                    best = {
                        "feature": feature,
                        "threshold_bin": int(threshold_bin),
                        "missing_left": bool(missing_left),
                        "gain": float(gain),
                    }
    return best


def split_rows_by_bin(X_binned, rows, split, *, missing_bin):
    """Apply a histogram split to row indices."""
    rows = np.asarray(rows, dtype=int)
    feature = int(split["feature"])
    bins = np.asarray(X_binned, dtype=np.int32)[rows, feature]
    feature_missing_bin = int(missing_bin(feature) if callable(missing_bin) else missing_bin)
    is_missing = bins == feature_missing_bin
    goes_left = (bins <= int(split["threshold_bin"])) & ~is_missing
    if split["missing_left"]:
        goes_left |= is_missing
    return rows[goes_left], rows[~goes_left], goes_left
