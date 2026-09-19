"""Correctness checks for the shared LightGBM core."""

import numpy as np

from classification.lightgbm_classification import build_histograms
from lightgbm_from_scratch.core.binning import create_quantile_bins, bin_matrix
from lightgbm_from_scratch.core.efb import build_efb_plan, build_efb_histograms
from lightgbm_from_scratch.core.goss import goss_sample
from lightgbm_from_scratch.objectives.binary import (
    binary_logloss_gradient_hessian,
    sigmoid,
)
from lightgbm_from_scratch.objectives.regression import squared_error_gradient_hessian
from regression.lightgbm_regression import LightGBMRegression


def test_squared_error_gradient_hessian_exact():
    y = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 17.0, 29.0])
    g, h = squared_error_gradient_hessian(y, pred)
    np.testing.assert_allclose(g, [2.0, -3.0, -1.0])
    np.testing.assert_allclose(h, np.ones(3))


def test_binary_gradient_matches_finite_difference():
    y = np.array([0.0, 1.0, 1.0])
    raw = np.array([-0.7, 0.2, 1.1])
    gradient, _ = binary_logloss_gradient_hessian(y, raw)
    eps = 1e-6

    def loss(score):
        p = np.clip(sigmoid(score), 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

    numerical = np.empty_like(raw)
    for i in range(raw.size):
        plus = raw.copy(); plus[i] += eps
        minus = raw.copy(); minus[i] -= eps
        numerical[i] = (loss(plus) - loss(minus)) / (2 * eps)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-6)


def test_goss_retains_largest_gradients_and_is_reproducible():
    gradients = np.arange(1.0, 31.0)
    rows1, weights1 = goss_sample(
        gradients, 0.2, 0.1, np.random.default_rng(7),
        count_mode="floor", weight_mode="rate_ratio", full_weight_array=True,
    )
    rows2, weights2 = goss_sample(
        gradients, 0.2, 0.1, np.random.default_rng(7),
        count_mode="floor", weight_mode="rate_ratio", full_weight_array=True,
    )
    assert set(range(24, 30)).issubset(set(rows1))
    np.testing.assert_array_equal(rows1, rows2)
    np.testing.assert_array_equal(weights1, weights2)


def test_efb_plan_detects_exact_mutual_exclusivity():
    X = np.array([
        [3.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [5.0, 0.0, 1.0],
        [0.0, 7.0, 1.0],
        [8.0, 0.0, 1.0],
        [0.0, 4.0, 1.0],
    ])
    cuts = create_quantile_bins(X, 8, quantile_style="centers")
    binned = bin_matrix(X, cuts, missing_bin="zero")
    plan = build_efb_plan(X, binned, missing_bin="zero")
    assert any(set(group) == {0, 1} for group in plan.groups)
    assert plan.applied


def test_classification_efb_histograms_equal_direct_histograms():
    X = np.array([
        [3.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [5.0, 0.0, np.nan],
        [0.0, 7.0, 1.0],
        [8.0, 0.0, 1.0],
        [0.0, 4.0, 1.0],
    ])
    cuts = create_quantile_bins(X, 8, quantile_style="centers")
    binned = bin_matrix(X, cuts, missing_bin="zero")
    plan = build_efb_plan(X, binned, missing_bin="zero")
    gradients = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    hessians = np.linspace(0.2, 0.7, 6)
    rows = np.array([0, 1, 2, 4, 5])
    weights = np.array([1.0, 2.0, 1.5, 1.0, 3.0])
    features = np.array([0, 1, 2])

    direct = build_histograms(
        binned, gradients, hessians, rows, weights, features, efb_plan=None
    )
    bundled = build_efb_histograms(
        plan, gradients, hessians, rows,
        weights=weights, features=features, include_count=True,
    )
    for feature in features:
        np.testing.assert_allclose(bundled[int(feature)], direct[int(feature)], atol=1e-12)


def test_regression_efb_histograms_equal_bruteforce():
    X = np.array([
        [3.0, 0.0],
        [0.0, 2.0],
        [5.0, 0.0],
        [0.0, 7.0],
        [8.0, 0.0],
        [0.0, 4.0],
    ])
    model = LightGBMRegression(n_estimators=1, max_bins=4, num_leaves=2)
    model._prepare_bins(X)
    assert model.efb_applied
    gradients = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    hessians = np.arange(1.0, 7.0)
    rows = np.arange(len(X))
    actual = model._histogram(rows, gradients, hessians)

    expected = np.zeros_like(actual)
    for feature in range(X.shape[1]):
        bins = model.feature_bins[:, feature]
        np.add.at(expected[feature, :, 0], bins, gradients)
        np.add.at(expected[feature, :, 1], bins, hessians)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_classifier_trains_with_efb_without_collapsing_feature_semantics():
    from classification.lightgbm_classification import LightGBMClassification

    rng = np.random.default_rng(12)
    n = 240
    selector = rng.integers(0, 2, size=n)
    X = np.zeros((n, 2), dtype=float)
    X[selector == 0, 0] = rng.uniform(0.5, 3.0, size=np.count_nonzero(selector == 0))
    X[selector == 1, 1] = rng.uniform(0.5, 3.0, size=np.count_nonzero(selector == 1))
    y = (X[:, 0] > 1.5).astype(int)

    model = LightGBMClassification(
        n_estimators=12,
        learning_rate=0.2,
        num_leaves=7,
        max_bins=16,
        min_child_samples=2,
        top_rate=0.3,
        other_rate=0.3,
        random_state=3,
    ).fit(X, y)
    assert model.efb_applied_
    assert any(set(group) == {0, 1} for group in model.feature_bundles_)
    proba = model.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.all(np.isfinite(proba))
    assert np.mean(model.predict(X) == y) > 0.85
