"""Early-stopping contracts shared by both public estimators."""

import numpy as np

from classification.lightgbm_classification import LightGBMClassification
from regression.lightgbm_regression import LightGBMRegression


def test_regression_early_stopping_keeps_best_iteration():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(120, 3))
    y = 2.0 * X[:, 0] - X[:, 1]
    model = LightGBMRegression(
        n_estimators=20,
        num_leaves=7,
        max_bins=8,
        min_gain_to_split=1e12,
        random_state=3,
    )
    model.fit(X[:80], y[:80], eval_set=(X[80:], y[80:]), early_stopping_rounds=3)
    assert model.best_iteration_ == 1
    assert model.n_estimators_ == 1
    assert len(model.evals_result_["validation"]["rmse"]) == 4
    assert np.isfinite(model.best_score_)


def test_classification_early_stopping_keeps_best_iteration_and_importance():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(160, 4))
    y = (X[:, 0] + 0.2 * X[:, 1] > 0).astype(int)
    model = LightGBMClassification(
        n_estimators=20,
        learning_rate=0.0,
        num_leaves=7,
        max_bins=8,
        min_split_gain=1e12,
        min_child_samples=2,
        top_rate=0.3,
        other_rate=0.3,
        random_state=4,
    )
    model.fit(X[:100], y[:100], eval_set=(X[100:], y[100:]), early_stopping_rounds=3)
    assert model.best_iteration_ == 1
    assert model.n_estimators_ == 1
    assert len(model.evals_result_["validation"]["binary_logloss"]) == 4
    assert model.get_feature_importance().shape == (4,)
    assert np.isfinite(model.best_score_)


def test_early_stopping_requires_validation_set():
    X = np.arange(40.0).reshape(20, 2)
    y_reg = np.linspace(0.0, 1.0, 20)
    y_cls = np.array([0, 1] * 10)

    try:
        LightGBMRegression(n_estimators=2).fit(
            X, y_reg, early_stopping_rounds=2
        )
    except ValueError as exc:
        assert "eval_set" in str(exc)
    else:
        raise AssertionError("Regression phải yêu cầu eval_set khi early stopping.")

    try:
        LightGBMClassification(
            n_estimators=2, min_child_samples=1, top_rate=0.3, other_rate=0.3
        ).fit(X, y_cls, early_stopping_rounds=2)
    except ValueError as exc:
        assert "eval_set" in str(exc)
    else:
        raise AssertionError("Classification phải yêu cầu eval_set khi early stopping.")
