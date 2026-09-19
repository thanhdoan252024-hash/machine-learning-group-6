"""Shared best-first (leaf-wise) tree engine."""

from __future__ import annotations

import heapq

import numpy as np

from .histogram import build_feature_histograms, histogram_subtraction
from .regularization import leaf_value
from .split import find_best_histogram_split, split_rows_by_bin


def _aligned_weights(rows, weights, n_samples):
    rows = np.asarray(rows, dtype=int)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size == rows.size:
        return weights
    if weights.size == n_samples:
        return weights[rows]
    raise ValueError("weights phải cùng chiều rows hoặc toàn bộ dataset.")


def build_leafwise_tree(
    X_binned,
    gradients,
    hessians,
    rows,
    weights,
    *,
    features,
    missing_bin,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=1,
    min_child_weight=0.0,
    min_gain=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    efb_plan=None,
):
    """Build one LightGBM-style tree by repeatedly splitting the best leaf.

    Returns ``(root_dict, gain_importance)``. The dict format intentionally
    matches the repository's historical regression tree contract.
    """
    X_binned = np.asarray(X_binned, dtype=np.int32)
    gradients = np.asarray(gradients, dtype=float).reshape(-1)
    hessians = np.asarray(hessians, dtype=float).reshape(-1)
    rows = np.asarray(rows, dtype=int).reshape(-1)
    features = np.asarray(features, dtype=int).reshape(-1)
    root_weights = _aligned_weights(rows, weights, X_binned.shape[0])

    def make_leaf(leaf_rows, leaf_weights, depth, histograms):
        g = float(np.sum(gradients[leaf_rows] * leaf_weights))
        h = float(np.sum(hessians[leaf_rows] * leaf_weights))
        return {
            "value": float(leaf_value(g, h, reg_alpha, reg_lambda)),
            "rows": leaf_rows,
            "weights": leaf_weights,
            "histogram": histograms,
            "gradient_sum": g,
            "hessian_sum": h,
            "depth": int(depth),
        }

    root_histogram = build_feature_histograms(
        X_binned,
        gradients,
        hessians,
        rows,
        root_weights,
        features,
        efb_plan=efb_plan,
        include_count=True,
        missing_bin=missing_bin,
    )
    root = make_leaf(rows, root_weights, 0, root_histogram)
    leaves = [root]
    queue = []
    counter = 0
    importance = np.zeros(X_binned.shape[1], dtype=float)

    def add_candidate(leaf):
        nonlocal counter
        if max_depth >= 0 and leaf["depth"] >= max_depth:
            return
        if leaf["rows"].size < 2 * max(1, int(min_child_samples)):
            return
        split = find_best_histogram_split(
            leaf["histogram"],
            features,
            missing_bin=missing_bin,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_gain=min_gain,
            min_child_samples=min_child_samples,
            min_child_weight=min_child_weight,
        )
        if split is not None:
            heapq.heappush(queue, (-split["gain"], counter, leaf, split))
            counter += 1

    add_candidate(root)
    while queue and len(leaves) < max(2, int(num_leaves)):
        _, _, leaf, split = heapq.heappop(queue)
        if "split" in leaf:
            continue
        left_rows, right_rows, mask = split_rows_by_bin(
            X_binned, leaf["rows"], split, missing_bin=missing_bin
        )
        if (
            left_rows.size < min_child_samples
            or right_rows.size < min_child_samples
            or left_rows.size == 0
            or right_rows.size == 0
        ):
            continue
        left_weights = leaf["weights"][mask]
        right_weights = leaf["weights"][~mask]
        left_h = float(np.sum(hessians[left_rows] * left_weights))
        right_h = float(np.sum(hessians[right_rows] * right_weights))
        if left_h < min_child_weight or right_h < min_child_weight:
            continue

        if left_rows.size <= right_rows.size:
            left_hist = build_feature_histograms(
                X_binned, gradients, hessians, left_rows, left_weights,
                features, efb_plan=efb_plan, include_count=True, missing_bin=missing_bin,
            )
            right_hist = histogram_subtraction(leaf["histogram"], left_hist)
        else:
            right_hist = build_feature_histograms(
                X_binned, gradients, hessians, right_rows, right_weights,
                features, efb_plan=efb_plan, include_count=True, missing_bin=missing_bin,
            )
            left_hist = histogram_subtraction(leaf["histogram"], right_hist)

        left = make_leaf(left_rows, left_weights, leaf["depth"] + 1, left_hist)
        right = make_leaf(right_rows, right_weights, leaf["depth"] + 1, right_hist)
        leaf["split"] = split
        leaf["left"] = left
        leaf["right"] = right
        leaves.remove(leaf)
        leaves.extend((left, right))
        importance[split["feature"]] += split["gain"]
        add_candidate(left)
        add_candidate(right)

    return root, importance


def predict_tree_dict(tree, X_binned, *, missing_bin):
    """Predict one tree using vectorized node-wise row routing.

    This is semantically identical to row-by-row traversal but avoids Python
    work per sample, which is important because boosting predicts over the
    whole training set after every newly-built tree.
    """
    X_binned = np.asarray(X_binned, dtype=np.int32)
    result = np.empty(X_binned.shape[0], dtype=float)
    stack = [(tree, np.arange(X_binned.shape[0], dtype=int))]
    while stack:
        node, rows = stack.pop()
        if rows.size == 0:
            continue
        if "split" not in node:
            result[rows] = float(node["value"])
            continue
        split = node["split"]
        feature = int(split["feature"])
        values = X_binned[rows, feature]
        feature_missing_bin = int(
            missing_bin(feature) if callable(missing_bin) else missing_bin
        )
        missing_mask = values == feature_missing_bin
        threshold_mask = values <= int(split["threshold_bin"])
        if bool(split["missing_left"]):
            left_mask = missing_mask | (~missing_mask & threshold_mask)
        else:
            left_mask = (~missing_mask) & threshold_mask
        stack.append((node["right"], rows[~left_mask]))
        stack.append((node["left"], rows[left_mask]))
    return result
