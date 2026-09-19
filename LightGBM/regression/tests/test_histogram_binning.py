"""Test histogram binning (binning + histogram gradient/hessian)."""

import numpy as np

from regression.lightgbm_regression import LightGBMRegression


def _make_model(max_bins=8):
    return LightGBMRegression(n_estimators=1, max_bins=max_bins, num_leaves=4)


def test_bins_shape_and_range():
    X = np.array(
        [[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0], [10.0, 12.0]]
    )
    model = _make_model(max_bins=4)
    model._prepare_bins(X)
    assert model.feature_bins.shape == X.shape
    assert model.feature_bins.min() >= 0
    assert model.feature_bins.max() <= model.max_bins


def test_bin_thresholds_are_sorted():
    X = np.random.default_rng(0).normal(size=(50, 3))
    model = _make_model(max_bins=8)
    model._prepare_bins(X)
    for thresholds in model.bin_thresholds:
        assert np.all(np.diff(thresholds) > 0)


def test_histogram_shape():
    X = np.random.default_rng(0).normal(size=(50, 3))
    model = _make_model(max_bins=8)
    model._prepare_bins(X)
    gradients = np.random.default_rng(1).normal(size=50)
    hessians = np.ones(50)
    hist = model._histogram(np.arange(50), gradients, hessians)
    assert hist.shape == (3, model.max_bins + 1, 2)


def test_histogram_gradient_total():
    X = np.random.default_rng(0).normal(size=(30, 2))
    model = _make_model(max_bins=8)
    model._prepare_bins(X)
    gradients = np.random.default_rng(1).normal(size=30)
    hessians = np.ones(30)
    rows = np.arange(30)
    hist = model._histogram(rows, gradients, hessians)
    for feature in range(2):
        assert np.isclose(hist[feature, :, 0].sum(), gradients.sum())


def test_histogram_hessian_total():
    X = np.random.default_rng(0).normal(size=(30, 2))
    model = _make_model(max_bins=8)
    model._prepare_bins(X)
    gradients = np.random.default_rng(1).normal(size=30)
    hessians = np.ones(30)
    rows = np.arange(30)
    hist = model._histogram(rows, gradients, hessians)
    for feature in range(2):
        assert np.isclose(hist[feature, :, 1].sum(), hessians.sum())


def test_histogram_subset_rows():
    X = np.random.default_rng(0).normal(size=(40, 2))
    model = _make_model(max_bins=8)
    model._prepare_bins(X)
    gradients = np.random.default_rng(1).normal(size=40)
    hessians = np.ones(40)
    subset = np.array([0, 5, 10, 15, 20])
    hist = model._histogram(subset, gradients, hessians)
    for feature in range(2):
        assert np.isclose(hist[feature, :, 0].sum(), gradients[subset].sum())
        assert np.isclose(hist[feature, :, 1].sum(), hessians[subset].sum())
