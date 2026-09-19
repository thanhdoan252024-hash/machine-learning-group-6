import numpy as np
import pytest

from regression.lightgbm_regression import LightGBMRegression
from classification.lightgbm_classification import LightGBMClassification


def _regression_data():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(120, 6))
    X[:, 4:] = 0.0
    X[:60, 4] = rng.normal(size=60)
    X[60:, 5] = rng.normal(size=60)
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=120)
    return X, y


def _classification_data():
    rng = np.random.default_rng(456)
    X = rng.normal(size=(240, 5))
    score = 1.8 * X[:, 0] - 1.2 * X[:, 1]
    y = (score > 1.5).astype(int)
    return X, y


def test_regression_can_disable_goss_and_efb():
    X, y = _regression_data()
    model = LightGBMRegression(
        n_estimators=5,
        num_leaves=4,
        max_bins=8,
        use_goss=False,
        use_efb=False,
        top_rate=0.0,
        other_rate=0.0,
    )
    model.fit(X, y)
    assert model.efb_plan_ is None
    assert model.efb_applied is False
    assert np.isfinite(model.predict(X)).all()


def test_regression_feature_fraction_validation_and_training():
    X, y = _regression_data()
    with pytest.raises(ValueError, match="feature_fraction"):
        LightGBMRegression(feature_fraction=1.2).fit(X, y)
    model = LightGBMRegression(
        n_estimators=4, num_leaves=4, max_bins=8, feature_fraction=0.5
    ).fit(X, y)
    assert model.feature_fraction == 0.5
    assert np.isfinite(model.predict(X[:5])).all()


def test_classification_can_disable_goss_and_efb():
    X, y = _classification_data()
    model = LightGBMClassification(
        n_estimators=6,
        num_leaves=4,
        max_bins=8,
        use_goss=False,
        use_efb=False,
        top_rate=0.0,
        other_rate=0.0,
        random_state=42,
    ).fit(X, y)
    assert model.efb_plan_ is None
    assert model.efb_applied_ is False
    assert np.isfinite(model.predict_proba(X)).all()


def test_classification_scale_pos_weight_validation():
    X, y = _classification_data()
    with pytest.raises(ValueError, match="scale_pos_weight"):
        LightGBMClassification(scale_pos_weight=0).fit(X, y)


def test_scale_pos_weight_changes_classifier_solution():
    X, y = _classification_data()
    common = dict(
        n_estimators=15,
        learning_rate=0.1,
        num_leaves=5,
        max_bins=16,
        random_state=42,
    )
    plain = LightGBMClassification(**common, scale_pos_weight=1.0).fit(X, y)
    weighted = LightGBMClassification(**common, scale_pos_weight=4.0).fit(X, y)
    p_plain = plain.predict_proba(X)[:, 1]
    p_weighted = weighted.predict_proba(X)[:, 1]
    assert not np.allclose(p_plain, p_weighted)
