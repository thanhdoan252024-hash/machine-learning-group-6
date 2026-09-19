"""Gradient-based One-Side Sampling (GOSS)."""

import numpy as np


def goss_sample(
    gradients,
    top_rate,
    other_rate,
    rng,
    *,
    count_mode="ceil",
    weight_mode="sample_ratio",
    full_weight_array=False,
):
    """Select large gradients and a random subset of the remaining examples.

    Parameters
    ----------
    count_mode:
        ``"ceil"`` matches the classification implementation; ``"floor"``
        preserves the historical regression implementation.
    weight_mode:
        ``"sample_ratio"`` uses ``len(pool) / len(sampled_small)``.
        ``"rate_ratio"`` uses ``(1-top_rate) / other_rate``.
    full_weight_array:
        If true, return one weight for every training row (legacy regression
        contract). Otherwise return weights aligned with the selected rows.
    """
    gradients = np.asarray(gradients, dtype=float).reshape(-1)
    n_samples = gradients.size
    if n_samples == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    if not (0 < top_rate < 1 and 0 < other_rate < 1 and top_rate + other_rate <= 1):
        raise ValueError("Tỉ lệ GOSS không hợp lệ.")

    counter = np.ceil if count_mode == "ceil" else np.floor
    n_top = max(1, int(counter(top_rate * n_samples)))
    n_other = max(1, int(counter(other_rate * n_samples)))

    order = np.argsort(np.abs(gradients))[::-1]
    large = order[:n_top]
    pool = order[n_top:]
    if pool.size > n_other:
        small = np.asarray(rng.choice(pool, n_other, replace=False), dtype=int)
    else:
        small = pool.astype(int, copy=False)
    rows = np.concatenate((large, small)).astype(int, copy=False)

    if weight_mode == "sample_ratio":
        small_weight = (pool.size / small.size) if small.size else 1.0
    elif weight_mode == "rate_ratio":
        small_weight = (1.0 - top_rate) / max(other_rate, 1e-12)
    else:
        raise ValueError("weight_mode không được hỗ trợ.")

    if full_weight_array:
        weights = np.ones(n_samples, dtype=float)
        if small.size:
            weights[small] = small_weight
    else:
        weights = np.ones(rows.size, dtype=float)
        if small.size:
            weights[large.size:] = small_weight
    return rows, weights
