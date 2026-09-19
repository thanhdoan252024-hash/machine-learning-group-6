"""Test tìm best split từ histogram."""

import numpy as np

from regression.lightgbm_regression import LightGBMRegression


def test_best_split_finds_separating_feature():
    # Feature 0 tách biệt hoàn hảo hai nhóm, feature 1 không tách được.
    X = np.array(
        [
            [0.1, 1.0],
            [0.2, 1.0],
            [0.3, 1.0],
            [0.4, 1.0],
            [0.5, 1.0],
            [5.1, 1.0],
            [5.2, 1.0],
            [5.3, 1.0],
            [5.4, 1.0],
            [5.5, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
    model = LightGBMRegression(
        max_bins=8, min_gain_to_split=0.0, reg_alpha=0.0, reg_lambda=0.0
    )
    model._prepare_bins(X)
    gradients = y - y.mean()
    hessians = np.ones(10)
    hist = model._histogram(np.arange(10), gradients, hessians)
    best = model._best_split(hist, gradients.sum(), hessians.sum())
    assert best is not None
    assert best["feature"] == 0
    assert best["gain"] > 0


def test_best_split_respects_min_gain_to_split():
    X = np.random.default_rng(0).normal(size=(30, 2))
    y = 2.0 * X[:, 0] + np.random.default_rng(1).normal(scale=0.1, size=30)
    model = LightGBMRegression(
        max_bins=8, min_gain_to_split=1e9, reg_alpha=0.0, reg_lambda=0.0
    )
    model._prepare_bins(X)
    gradients = y - y.mean()
    hessians = np.ones(30)
    hist = model._histogram(np.arange(30), gradients, hessians)
    best = model._best_split(hist, gradients.sum(), hessians.sum())
    # gain bị trừ min_gain_to_split rất lớn nên best_split có gain rất âm
    assert best is not None
    assert best["gain"] < 0


def test_no_split_when_no_gain():
    # y hằng số -> gradient toàn 0 -> không có gain -> cây chỉ 1 lá
    X = np.random.default_rng(0).normal(size=(20, 2))
    y = np.ones(20)
    model = LightGBMRegression(
        max_bins=8, min_gain_to_split=0.0, reg_alpha=0.0, reg_lambda=0.0
    )
    model._prepare_bins(X)
    gradients = y - y.mean()  # toàn 0
    hessians = np.ones(20)
    tree = model._best_first_tree(np.arange(20), gradients, hessians)
    assert "split" not in tree
