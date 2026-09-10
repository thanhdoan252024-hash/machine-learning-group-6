"""Test cơ chế leaf-wise (best-first) của việc xây cây."""

import numpy as np

from regression.lightgbm_regression import LightGBMRegression
from regression.tests._helpers import count_leaves, make_regression_data


def _build_tree(model, X, y, max_bins=16):
    model._prepare_bins(X)
    gradients = y - y.mean()
    hessians = np.ones(len(y))
    return model._best_first_tree(np.arange(len(y)), gradients, hessians)


def test_leaf_wise_respects_num_leaves():
    X, y = make_regression_data(n=300, noise=0.2)
    model = LightGBMRegression(
        n_estimators=1, num_leaves=8, max_bins=16, min_data_in_leaf=1
    )
    tree = _build_tree(model, X, y)
    assert count_leaves(tree) <= 8


def test_leaf_wise_root_splits_with_signal():
    X, y = make_regression_data(n=300, noise=0.1)
    model = LightGBMRegression(
        n_estimators=1, num_leaves=8, max_bins=16, min_data_in_leaf=1
    )
    tree = _build_tree(model, X, y)
    # Dữ liệu có tín hiệu mạnh nên cây phải tách ở gốc
    assert "split" in tree


def test_leaf_wise_is_valid_binary_tree():
    X, y = make_regression_data(n=300, noise=0.2)
    model = LightGBMRegression(
        n_estimators=1, num_leaves=8, max_bins=16, min_data_in_leaf=1
    )
    tree = _build_tree(model, X, y)
    # Mỗi nút có split phải có đúng 2 con (left, right)
    stack = [tree]
    while stack:
        node = stack.pop()
        if "split" in node:
            assert "left" in node and "right" in node
            stack.append(node["left"])
            stack.append(node["right"])


def test_leaf_wise_can_be_unbalanced():
    """Leaf-wise có thể tạo cây không cân bằng (độ sâu khác nhau giữa nhánh)."""
    X, y = make_regression_data(n=500, noise=0.05)
    model = LightGBMRegression(
        n_estimators=1, num_leaves=10, max_bins=16, min_data_in_leaf=1
    )
    tree = _build_tree(model, X, y)

    depths = []

    def collect_depths(node, depth):
        if "split" not in node:
            depths.append(depth)
            return
        collect_depths(node["left"], depth + 1)
        collect_depths(node["right"], depth + 1)

    collect_depths(tree, 0)
    # Nếu cây có > 1 lá thì độ sâu các lá nên có sự khác biệt (leaf-wise)
    if len(depths) > 1:
        assert len(set(depths)) > 1
