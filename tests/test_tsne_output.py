"""Test kết quả giảm chiều cuối cùng của TSNE trong core/tsne_numpy.py.

Kiểm tra shape, số lượng mẫu, tính hữu hạn, không bị collapse, và hành vi
đúng của fit() / fit_transform() theo API thực tế.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Đưa thư mục gốc (D:/t-SNE) vào sys.path để import được `core.tsne_numpy`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tsne_numpy import TSNE  # noqa: E402


def make_data(n_per_class: int = 20, n_features: int = 5, seed: int = 0) -> np.ndarray:
    """Dataset nhỏ tự sinh: 2 cụm Gaussian tách biệt (40 mẫu, 5 đặc trưng)."""
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(-3.0, 1.0, (n_per_class, n_features))
    cluster_b = rng.normal(3.0, 1.0, (n_per_class, n_features))
    return np.vstack([cluster_a, cluster_b])


def make_model(**kwargs) -> TSNE:
    """TSNE với hyperparameter nhỏ để test nhanh."""
    defaults = dict(n_components=2, perplexity=10.0, n_iter=200, random_state=42)
    defaults.update(kwargs)
    return TSNE(**defaults)


# ---------------------------------------------------------------------------
# Shape & số lượng mẫu
# ---------------------------------------------------------------------------
def test_fit_transform_shape_two_components():
    X = make_data()
    Y = make_model().fit_transform(X)
    # n_components=2 => output (n_samples, 2)
    assert Y.shape == (X.shape[0], 2)


def test_sample_count_unchanged():
    X = make_data()
    Y = make_model().fit_transform(X)
    assert Y.shape[0] == X.shape[0]


def test_default_n_components_is_two():
    X = make_data()
    Y = TSNE(perplexity=10.0, n_iter=200, random_state=42).fit_transform(X)
    assert Y.shape[1] == 2


# ---------------------------------------------------------------------------
# Tính hữu hạn & không collapse
# ---------------------------------------------------------------------------
def test_no_nan_or_inf():
    X = make_data()
    Y = make_model().fit_transform(X)
    assert np.all(np.isfinite(Y))


def test_embedding_not_collapsed():
    X = make_data()
    Y = make_model().fit_transform(X)
    # Không được collapse thành các điểm giống hệt nhau:
    # ít nhất 2 điểm phân biệt và phương sai > 0.
    assert np.unique(Y, axis=0).shape[0] > 1
    assert Y.std() > 0.0


# ---------------------------------------------------------------------------
# Hành vi fit() / fit_transform()
# ---------------------------------------------------------------------------
def test_fit_returns_self_and_sets_embedding():
    X = make_data()
    model = make_model()
    result = model.fit(X)
    # fit() trả về chính đối tượng
    assert result is model
    # embedding được lưu trong thuộc tính embedding_
    assert hasattr(model, "embedding_")
    assert model.embedding_.shape == (X.shape[0], 2)


def test_fit_transform_matches_fit_embedding():
    X = make_data()
    # fit_transform phải cho kết quả giống fit().embedding_
    Y = make_model().fit_transform(X)
    model = make_model().fit(X)
    np.testing.assert_allclose(model.embedding_, Y)


def test_fit_sets_expected_attributes():
    X = make_data()
    model = make_model().fit(X)
    # Các thuộc tính kết quả mà source đảm bảo sau khi fit
    assert model.P_.shape == (X.shape[0], X.shape[0])
    assert model.betas_.shape == (X.shape[0],)
    assert np.isfinite(model.kl_divergence_)
    assert model.n_iter_ >= 1


def test_fit_tranform_alias_exists():
    X = make_data()
    # Source có alias fit_tranform (thiếu chữ s) trỏ về fit_transform
    Y = make_model().fit_tranform(X)
    assert Y.shape == (X.shape[0], 2)
