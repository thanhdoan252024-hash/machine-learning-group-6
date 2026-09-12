"""Test các điều kiện dừng khi xây cây: num_leaves, max_depth,
min_data_in_leaf, min_gain_to_split."""

import numpy as np

from regression.lightgbm_regression import LightGBMRegression
from regression.tests._helpers import (
    count_leaves,
    make_regression_data,
    max_depth,
    min_leaf_size,
)


def _build_tree(model, X, y):
    model._prepare_bins(X)
    gradients = y - y.mean()
    hessians = np.ones(len(y))
    return model._best_first_tree(np.arange(len(y)), gradients, hessians)


def test_num_leaves_limit():
    X, y = make_regression_data(n=300, noise=0.2)
    model = LightGBMRegression(
        n_estimators=1, num_leaves=6, max_bins=16, min_data_in_leaf=1
    )
    tree = _build_tree(model, X, y)
    assert count_leaves(tree) <= 6


def test_max_depth_limit():
    X, y = make_regression_data(n=300, noise=0.2)
    model = LightGBMRegression(
        n_estimators=1, num_leaves=100, max_depth=3, max_bins=16, min_data_in_leaf=1
    )
    tree = _build_tree(model, X, y)
    assert max_depth(tree) <= 3


def test_min_data_in_leaf_limit():
    X, y = make_regression_data(n=300, noise=0.2)
    model = LightGBMRegression(
        n_estimators=1, num_leaves=100, max_bins=16, min_data_in_leaf=50
    )
    tree = _build_tree(model, X, y)
    assert min_leaf_size(tree) >= 50


def test_min_gain_to_split_prevents_splitting():
    X, y = make_regression_data(n=300, noise=0.2)
    model = LightGBMRegression(
        n_estimators=1,
        num_leaves=100,
        max_bins=16,
        min_data_in_leaf=1,
        min_gain_to_split=1e9,
    )
    tree = _build_tree(model, X, y)
    assert "split" not in tree


def test_num_leaves_two_means_single_split():
    X, y = make_regression_data(n=300, noise=0.2)
    model = LightGBMRegression(
        n_estimators=1, num_leaves=2, max_bins=16, min_data_in_leaf=1
    )
    tree = _build_tree(model, X, y)
    assert count_leaves(tree) == 2


def test_min_data_in_leaf_validation():
    model = LightGBMRegression(min_data_in_leaf=0)
    X = np.random.default_rng(0).normal(size=(10, 2))
    y = np.random.default_rng(1).normal(size=10)
    try:
        model.fit(X, y)
        assert model.min_data_in_leaf >= 1  # bị ép về tối thiểu 1
    except ValueError:
        pass
