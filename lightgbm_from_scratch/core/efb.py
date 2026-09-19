"""Correctness-first Exclusive Feature Bundling for histogram construction.

The previous project versions collapsed several original features into a new
feature and then split directly on that encoded feature. That can change split
semantics. This module instead bundles only the *histogram construction* pass
and reconstructs an exact histogram for each original feature.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class EFBPlan:
    groups: List[List[int]]
    bundle_codes: np.ndarray
    offsets: Dict[int, int]
    valid_bin_counts: np.ndarray
    zero_bins: np.ndarray
    missing_bins: np.ndarray
    binned_data: np.ndarray
    exact: bool = True

    @property
    def applied(self):
        return any(len(group) > 1 for group in self.groups)


def _find_exact_exclusive_groups(raw_X):
    active = np.isfinite(raw_X) & (raw_X != 0)
    groups = []
    for feature in range(raw_X.shape[1]):
        for group in groups:
            occupied = np.any(active[:, group], axis=1)
            if not np.any(occupied & active[:, feature]):
                group.append(feature)
                break
        else:
            groups.append([feature])
    return groups, active


def build_efb_plan(raw_X, binned_X, *, missing_bin="zero", max_bins=None):
    """Create an exact-conflict-free EFB plan from *raw training data*.

    Sparsity is detected before binning, so a raw value of zero remains an
    inactive value even though its quantized bin id can be non-zero.
    """
    raw_X = np.asarray(raw_X, dtype=float)
    binned_X = np.asarray(binned_X, dtype=np.int32)
    if raw_X.shape != binned_X.shape or raw_X.ndim != 2:
        raise ValueError("raw_X và binned_X phải là hai ma trận cùng kích thước.")

    groups, active = _find_exact_exclusive_groups(raw_X)
    n_features = raw_X.shape[1]

    if missing_bin == "zero":
        missing_bins = np.zeros(n_features, dtype=np.int32)
        valid_bin_counts = np.array(
            [int(binned_X[:, j].max()) for j in range(n_features)], dtype=np.int32
        )
        # At least one valid bin is needed for a constant/all-missing feature.
        valid_bin_counts = np.maximum(valid_bin_counts, 1)
        local_bin = lambda bins: bins - 1
    elif missing_bin == "last":
        if max_bins is None:
            raise ValueError("Cần max_bins khi missing_bin='last'.")
        missing_bins = np.full(n_features, int(max_bins), dtype=np.int32)
        valid_bin_counts = np.full(n_features, int(max_bins), dtype=np.int32)
        local_bin = lambda bins: bins
    else:
        raise ValueError("missing_bin không hợp lệ.")

    zero_bins = np.empty(n_features, dtype=np.int32)
    for j in range(n_features):
        zero_rows = np.flatnonzero(np.isfinite(raw_X[:, j]) & (raw_X[:, j] == 0))
        if zero_rows.size:
            zero_bins[j] = int(binned_X[zero_rows[0], j])
        else:
            # Infer the bin that zero would occupy from neighboring observed
            # bins is impossible without cuts. Use search-equivalent behavior
            # via the closest finite value only as a fallback; callers should
            # normally include sparse zeros when requesting EFB.
            finite_rows = np.flatnonzero(np.isfinite(raw_X[:, j]))
            if finite_rows.size == 0:
                zero_bins[j] = 1 if missing_bin == "zero" else 0
            else:
                nearest = finite_rows[np.argmin(np.abs(raw_X[finite_rows, j]))]
                zero_bins[j] = int(binned_X[nearest, j])

    offsets = {}
    bundle_codes = np.zeros((raw_X.shape[0], len(groups)), dtype=np.int32)
    for bundle_idx, group in enumerate(groups):
        offset = 0
        occupied = np.zeros(raw_X.shape[0], dtype=bool)
        for feature in group:
            offsets[feature] = offset
            mask = active[:, feature]
            if np.any(occupied & mask):
                raise RuntimeError("EFB plan không còn exclusive; đây là lỗi nội bộ.")
            bins = binned_X[mask, feature]
            local = local_bin(bins)
            bundle_codes[mask, bundle_idx] = 1 + offset + local
            occupied |= mask
            offset += int(valid_bin_counts[feature])

    return EFBPlan(
        groups=groups,
        bundle_codes=bundle_codes,
        offsets=offsets,
        valid_bin_counts=valid_bin_counts,
        zero_bins=zero_bins,
        missing_bins=missing_bins,
        binned_data=binned_X,
    )


def build_efb_histograms(
    plan: EFBPlan,
    gradients,
    hessians,
    rows,
    *,
    weights=None,
    features: Iterable[int] | None = None,
    include_count=True,
):
    """Build exact per-feature histograms while scanning one code per bundle.

    Returns a dict ``feature -> [gradient, hessian, count]`` (or the first two
    rows if ``include_count=False``). Inactive-zero and missing contributions
    are reconstructed from totals, preserving original feature semantics.
    """
    gradients = np.asarray(gradients, dtype=float)
    hessians = np.asarray(hessians, dtype=float)
    rows = np.asarray(rows, dtype=int)
    if weights is None:
        weights = np.ones(rows.size, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != rows.shape:
            raise ValueError("weights phải cùng chiều với rows.")

    requested = set(range(plan.binned_data.shape[1]) if features is None else map(int, features))
    weighted_g = gradients[rows] * weights
    weighted_h = hessians[rows] * weights
    total_g = float(weighted_g.sum())
    total_h = float(weighted_h.sum())
    total_count = float(rows.size)
    result = {}

    for bundle_idx, group in enumerate(plan.groups):
        selected = [feature for feature in group if feature in requested]
        if not selected:
            continue
        codes = plan.bundle_codes[rows, bundle_idx]
        width = 1 + sum(int(plan.valid_bin_counts[f]) for f in group)
        bundle_g = np.bincount(codes, weights=weighted_g, minlength=width)
        bundle_h = np.bincount(codes, weights=weighted_h, minlength=width)
        bundle_c = np.bincount(codes, minlength=width).astype(float)

        for feature in selected:
            if plan.missing_bins[feature] == 0:
                n_bins = int(plan.valid_bin_counts[feature]) + 1
                valid_start = 1
            else:
                n_bins = int(plan.missing_bins[feature]) + 1
                valid_start = 0
            hist = np.zeros((3, n_bins), dtype=float)

            offset = int(plan.offsets[feature])
            count = int(plan.valid_bin_counts[feature])
            segment = slice(1 + offset, 1 + offset + count)
            active_g = bundle_g[segment]
            active_h = bundle_h[segment]
            active_c = bundle_c[segment]
            hist[0, valid_start:valid_start + count] += active_g
            hist[1, valid_start:valid_start + count] += active_h
            hist[2, valid_start:valid_start + count] += active_c

            missing_mask = plan.binned_data[rows, feature] == plan.missing_bins[feature]
            missing_g = float(weighted_g[missing_mask].sum())
            missing_h = float(weighted_h[missing_mask].sum())
            missing_c = float(np.count_nonzero(missing_mask))
            missing_idx = int(plan.missing_bins[feature])
            hist[0, missing_idx] += missing_g
            hist[1, missing_idx] += missing_h
            hist[2, missing_idx] += missing_c

            zero_g = total_g - float(active_g.sum()) - missing_g
            zero_h = total_h - float(active_h.sum()) - missing_h
            zero_c = total_count - float(active_c.sum()) - missing_c
            zero_idx = int(plan.zero_bins[feature])
            hist[0, zero_idx] += zero_g
            hist[1, zero_idx] += zero_h
            hist[2, zero_idx] += zero_c

            # Numerical subtraction can create tiny floating residuals only.
            hist[np.abs(hist) < 1e-15] = 0.0
            result[feature] = hist if include_count else hist[:2]

    return result
