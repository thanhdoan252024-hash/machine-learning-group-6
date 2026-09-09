"""Test các hàm cốt lõi của t-SNE trong core/tsne_numpy.py.

Chỉ kiểm tra các hàm module-level (distance, perplexity/entropy, P, Q,
gradient, KL divergence) dựa trên đúng tên hàm đã đọc từ source.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Đưa thư mục gốc (D:/t-SNE) vào sys.path để import được `core.tsne_numpy`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tsne_numpy import (  # noqa: E402
    binary_search_beta,
    conditional_probabilities,
    entropy_and_probabilities,
    gradient_descent_step,
    initialize_embedding,
    kl_divergence,
    squared_euclidean_distances,
    student_t_probabilities,
    symmetric_probabilities,
    tsne_gradient,
)


# ---------------------------------------------------------------------------
# squared_euclidean_distances
# ---------------------------------------------------------------------------
def test_squared_euclidean_distances_known_values():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    D = squared_euclidean_distances(X)
    assert D.shape == (2, 2)
    # ||(0,0) - (3,4)||^2 = 25
    assert np.isclose(D[0, 1], 25.0)
    assert np.isclose(D[1, 0], 25.0)
    # đường chéo phải bằng 0
    np.testing.assert_allclose(np.diag(D), 0.0)


def test_squared_euclidean_distances_symmetric():
    X = np.random.default_rng(0).normal(size=(7, 4))
    D = squared_euclidean_distances(X)
    assert np.allclose(D, D.T)
    assert np.all(D >= 0.0)


# ---------------------------------------------------------------------------
# entropy_and_probabilities / binary_search_beta (perplexity)
# ---------------------------------------------------------------------------
def test_entropy_and_probabilities_sum_to_one():
    d = np.array([0.0, 1.0, 2.0, 3.0])
    entropy, probs = entropy_and_probabilities(d, beta=1.0)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0.0)
    assert entropy >= 0.0


def test_higher_beta_gives_lower_entropy():
    d = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    e_low, _ = entropy_and_probabilities(d, beta=0.1)
    e_high, _ = entropy_and_probabilities(d, beta=10.0)
    # beta lớn => phân phối tập trung hơn => entropy nhỏ hơn
    assert e_high < e_low


def test_binary_search_beta_matches_perplexity():
    d = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    perplexity = 3.0
    beta, probs = binary_search_beta(d, perplexity)
    # H(probabilities) phải xấp xỉ ln(perplexity)
    entropy, _ = entropy_and_probabilities(d, beta)
    assert np.isclose(entropy, np.log(perplexity), atol=1e-4)
    assert np.isclose(probs.sum(), 1.0)


# ---------------------------------------------------------------------------
# conditional_probabilities / symmetric_probabilities (P)
# ---------------------------------------------------------------------------
def test_conditional_probabilities_rows_sum_to_one():
    rng = np.random.default_rng(1)
    D = squared_euclidean_distances(rng.normal(size=(6, 3)))
    P_cond, betas = conditional_probabilities(D, perplexity=3.0)
    assert P_cond.shape == (6, 6)
    assert betas.shape == (6,)
    # p(i|i) = 0
    np.testing.assert_allclose(np.diag(P_cond), 0.0)
    # mỗi hàng (trừ đường chéo) phải cộng tới 1
    np.testing.assert_allclose(P_cond.sum(axis=1), 1.0, atol=1e-6)


def test_symmetric_probabilities():
    rng = np.random.default_rng(2)
    D = squared_euclidean_distances(rng.normal(size=(6, 3)))
    P_cond, _ = conditional_probabilities(D, perplexity=3.0)
    P = symmetric_probabilities(P_cond)
    # đối xứng, tổng bằng 1, đường chéo bằng 0
    assert np.allclose(P, P.T)
    np.testing.assert_allclose(P.sum(), 1.0)
    np.testing.assert_allclose(np.diag(P), 0.0)


# ---------------------------------------------------------------------------
# student_t_probabilities (Q)
# ---------------------------------------------------------------------------
def test_student_t_probabilities():
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(5, 2))
    Q, numerator = student_t_probabilities(Y)
    # Q là phân phối: tổng 1, đường chéo 0
    np.testing.assert_allclose(Q.sum(), 1.0)
    np.testing.assert_allclose(np.diag(Q), 0.0)
    # numerator = 1 / (1 + ||y_i - y_j||^2)
    D = squared_euclidean_distances(Y)
    expected = 1.0 / (1.0 + D)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(numerator, expected)


# ---------------------------------------------------------------------------
# kl_divergence
# ---------------------------------------------------------------------------
def test_kl_divergence_zero_when_equal():
    P = np.array([[0.0, 0.5], [0.5, 0.0]])
    assert np.isclose(kl_divergence(P, P), 0.0)


def test_kl_divergence_non_negative():
    rng = np.random.default_rng(4)
    D = squared_euclidean_distances(rng.normal(size=(6, 3)))
    P_cond, _ = conditional_probabilities(D, perplexity=3.0)
    P = symmetric_probabilities(P_cond)
    Q, _ = student_t_probabilities(rng.normal(size=(6, 2)))
    assert kl_divergence(P, Q) >= 0.0


# ---------------------------------------------------------------------------
# initialize_embedding / tsne_gradient / gradient_descent_step
# ---------------------------------------------------------------------------
def test_initialize_embedding_shape_and_finite():
    rng = np.random.default_rng(5)
    Y = initialize_embedding(n=8, n_components=2, rng=rng)
    assert Y.shape == (8, 2)
    assert np.all(np.isfinite(Y))


def test_tsne_gradient_shape_and_finite():
    rng = np.random.default_rng(6)
    D = squared_euclidean_distances(rng.normal(size=(6, 3)))
    P_cond, _ = conditional_probabilities(D, perplexity=3.0)
    P = symmetric_probabilities(P_cond)
    Y = rng.normal(size=(6, 2))
    Q, numerator = student_t_probabilities(Y)
    grad = tsne_gradient(P, Q, Y, numerator)
    assert grad.shape == Y.shape
    assert np.all(np.isfinite(grad))


def test_gradient_descent_step_updates_state():
    rng = np.random.default_rng(7)
    Y = rng.normal(size=(6, 2))
    velocity = np.zeros_like(Y)
    gains = np.ones_like(Y)
    grad = rng.normal(size=Y.shape)
    Y2, v2, g2 = gradient_descent_step(
        Y, velocity, gains, grad, learning_rate=200.0,
        momentum=0.5, min_gain=0.01)
    assert Y2.shape == Y.shape
    assert v2.shape == velocity.shape
    assert g2.shape == gains.shape
    assert np.all(np.isfinite(Y2))
    # sau một bước, embedding phải thay đổi so với trước
    assert not np.allclose(Y2, Y)
