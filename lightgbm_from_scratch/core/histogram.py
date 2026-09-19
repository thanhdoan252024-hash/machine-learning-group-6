"""Shared histogram construction for the educational LightGBM engine."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .efb import EFBPlan, build_efb_histograms


def histogram_subtraction(parent_histogram, child_histogram):
    """Return sibling histograms using LightGBM's histogram subtraction trick."""
    return {
        int(feature): np.asarray(parent_histogram[feature]) - np.asarray(child_histogram[feature])
        for feature in parent_histogram
    }


def build_feature_histograms(
    X_binned,
    gradients,
    hessians,
    rows,
    weights,
    features: Iterable[int],
    *,
    efb_plan: EFBPlan | None = None,
    include_count: bool = True,
    missing_bin=None,
):
    """Build per-feature gradient/Hessian/(count) histograms.

    ``weights`` is aligned with ``rows``. When an EFB plan is supplied, the
    bundled scan is used while exact original-feature histograms are rebuilt.
    """
    X_binned = np.asarray(X_binned, dtype=np.int32)
    gradients = np.asarray(gradients, dtype=float).reshape(-1)
    hessians = np.asarray(hessians, dtype=float).reshape(-1)
    rows = np.asarray(rows, dtype=int).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    features = np.asarray(list(features), dtype=int).reshape(-1)

    if X_binned.ndim != 2:
        raise ValueError("X_binned phải là ma trận hai chiều.")
    if gradients.shape != hessians.shape or gradients.size != X_binned.shape[0]:
        raise ValueError("Gradient/Hessian phải khớp số dòng của X_binned.")
    if rows.shape != weights.shape:
        raise ValueError("weights phải cùng chiều với rows.")
    if rows.size and (rows.min() < 0 or rows.max() >= X_binned.shape[0]):
        raise ValueError("rows chứa chỉ số ngoài phạm vi dữ liệu.")

    if efb_plan is not None and efb_plan.applied:
        return build_efb_histograms(
            efb_plan,
            gradients,
            hessians,
            rows,
            weights=weights,
            features=features,
            include_count=include_count,
        )

    weighted_g = gradients[rows] * weights
    weighted_h = hessians[rows] * weights
    histograms = {}
    for feature in features:
        bins = X_binned[rows, feature]
        if efb_plan is not None:
            if int(efb_plan.missing_bins[feature]) == 0:
                n_bins = int(efb_plan.valid_bin_counts[feature]) + 1
            else:
                n_bins = int(efb_plan.missing_bins[feature]) + 1
        else:
            n_bins = int(X_binned[:, feature].max()) + 1
            if missing_bin is not None:
                feature_missing_bin = int(
                    missing_bin(feature) if callable(missing_bin) else missing_bin
                )
                n_bins = max(n_bins, feature_missing_bin + 1)
        rows_out = [
            np.bincount(bins, weights=weighted_g, minlength=n_bins),
            np.bincount(bins, weights=weighted_h, minlength=n_bins),
        ]
        if include_count:
            rows_out.append(np.bincount(bins, minlength=n_bins).astype(float))
        histograms[int(feature)] = np.vstack(rows_out)
    return histograms
