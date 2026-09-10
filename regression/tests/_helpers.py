"""Hàm tiện ích dùng chung cho các test."""

import numpy as np


def make_regression_data(n=200, n_features=3, seed=0, noise=0.2):
    """Tạo dữ liệu regression giả (không dùng sklearn)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = (
        3.0 * X[:, 0]
        - 2.0 * X[:, 1]
        + 1.0 * X[:, 2]
        + rng.normal(scale=noise, size=n)
    )
    return X, y


def count_leaves(node):
    """Đếm số lá của cây (node là dict từ _best_first_tree)."""
    if "split" not in node:
        return 1
    return count_leaves(node["left"]) + count_leaves(node["right"])


def max_depth(node, depth=0):
    """Tính độ sâu tối đa của cây."""
    if "split" not in node:
        return depth
    return max(
        max_depth(node["left"], depth + 1),
        max_depth(node["right"], depth + 1),
    )


def min_leaf_size(node):
    """Tính số mẫu tối thiểu trong một lá của cây."""
    if "split" not in node:
        return node["rows"].size
    return min(min_leaf_size(node["left"]), min_leaf_size(node["right"]))
