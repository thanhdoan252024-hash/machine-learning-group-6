"""t-SNE thuần NumPy; mọi phép tính nằm ngoài lớp TSNE."""
from __future__ import annotations
import numpy as np


def squared_euclidean_distances(X: np.ndarray) -> np.ndarray:
    """D_ij=||x_i-x_j||²=||x_i||²+||x_j||²-2<x_i,x_j>."""
    norms = np.sum(X * X, axis=1)
    D = np.maximum(norms[:, None] + norms[None, :] - 2.0 * X @ X.T, 0.0)
    np.fill_diagonal(D, 0.0)
    return D


def entropy_and_probabilities(distances: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    """p_j=exp(-beta*d_j)/sum_k exp(-beta*d_k); H=-sum_j p_j ln(p_j).

    beta=1/(2*sigma²), perplexity=exp(H). Dịch logits trước exp để ổn định số.
    """
    logits = -beta * distances
    weights = np.exp(logits - np.max(logits))
    probabilities = weights / weights.sum()
    positive = probabilities > 0
    entropy = -np.sum(probabilities[positive] * np.log(probabilities[positive]))
    return float(entropy), probabilities


def binary_search_beta(distances: np.ndarray, perplexity: float,
                       tolerance: float = 1e-5, max_iter: int = 50) -> tuple[float, np.ndarray]:
    """Tìm beta sao cho H gần ln(perplexity); H giảm đơn điệu khi beta tăng."""
    target, beta, lower, upper = np.log(perplexity), 1.0, -np.inf, np.inf
    probabilities = np.empty_like(distances)
    for _ in range(max_iter):
        entropy, probabilities = entropy_and_probabilities(distances, beta)
        difference = entropy - target
        if abs(difference) <= tolerance:
            break
        if difference > 0:  # phân phối còn quá rộng: tăng beta, giảm sigma
            lower = beta
            beta = beta * 2 if np.isinf(upper) else (beta + upper) / 2
        else:
            upper = beta
            beta = beta / 2 if np.isinf(lower) else (beta + lower) / 2
    return beta, probabilities


def conditional_probabilities(D: np.ndarray, perplexity: float,
                              tolerance: float = 1e-5, max_iter: int = 50
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Tính toàn bộ p(j|i), với p(i|i)=0, và beta riêng của từng điểm."""
    n = D.shape[0]
    P_cond, betas = np.zeros_like(D), np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        betas[i], P_cond[i, mask] = binary_search_beta(
            D[i, mask], perplexity, tolerance, max_iter)
    return P_cond, betas


def symmetric_probabilities(P_cond: np.ndarray) -> np.ndarray:
    """p_ij=(p(j|i)+p(i|j))/(2N), vì vậy P đối xứng và sum(P)=1."""
    P = (P_cond + P_cond.T) / (2.0 * len(P_cond))
    np.fill_diagonal(P, 0.0)
    return P


def initialize_embedding(n: int, n_components: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Khởi tạo y_i ~ N(0, 10^-4²), gần gốc tọa độ."""
    return rng.normal(0.0, 1e-4, (n, n_components))


def student_t_probabilities(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """a_ij=(1+||y_i-y_j||²)^-1; q_ij=a_ij/sum_{k!=l}a_kl.

    Student-t một bậc tự do có đuôi dày, giúp giảm vấn đề chen chúc (crowding).
    """
    numerator = 1.0 / (1.0 + squared_euclidean_distances(Y))
    np.fill_diagonal(numerator, 0.0)
    return numerator / numerator.sum(), numerator


def kl_divergence(P: np.ndarray, Q: np.ndarray, epsilon: float = 1e-12) -> float:
    """KL(P||Q)=sum_{i!=j} p_ij*ln(p_ij/q_ij)."""
    mask = P > 0
    return float(np.sum(P[mask] * np.log(P[mask] / np.maximum(Q[mask], epsilon))))


def tsne_gradient(P: np.ndarray, Q: np.ndarray, Y: np.ndarray,
                  numerator: np.ndarray) -> np.ndarray:
    """dC/dy_i=4*sum_j (p_ij-q_ij)(y_i-y_j)/(1+||y_i-y_j||²).

    Với W=(P-Q)*numerator: gradient=4*(sum_hàng(W)*Y-W@Y).
    """
    W = (P - Q) * numerator
    return 4.0 * (W.sum(axis=1)[:, None] * Y - W @ Y)


def gradient_descent_step(Y: np.ndarray, velocity: np.ndarray, gradient: np.ndarray,
                          gains: np.ndarray, learning_rate: float, momentum: float,
                          min_gain: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Một bước gradient descent với momentum và adaptive gains gốc.

    g_t = g_(t-1)+0.2 nếu sign(gradient_t) != sign(velocity_(t-1)),
          0.8*g_(t-1) trong trường hợp còn lại.
    Sau đó: g_t=max(g_t,min_gain),
    velocity_t=momentum*velocity_(t-1)-learning_rate*g_t*gradient_t.
    """
    changed_direction = np.sign(gradient) != np.sign(velocity)
    gains = np.where(changed_direction, gains + 0.2, gains * 0.8)
    gains = np.maximum(gains, min_gain)
    velocity = momentum * velocity - learning_rate * gains * gradient
    Y = Y + velocity
    Y -= Y.mean(axis=0, keepdims=True)
    return Y, velocity, gains


class TSNE:
    """t-Distributed Stochastic Neighbor Embedding viết thuần bằng NumPy.

    Bản exact t-SNE này cần O(N²) bộ nhớ. Các hyperparameter đều là thuộc tính;
    các thuộc tính có hậu tố ``_`` được tạo sau khi fit.
    """
    def __init__(self, n_components: int = 2, perplexity: float = 30.0,
                 learning_rate: float = 200.0, n_iter: int = 1000,
                 early_exaggeration: float = 12.0, early_exaggeration_iter: int = 250,
                 momentum: float = 0.5, final_momentum: float = 0.8,
                 min_gain: float = 0.01,
                 patience: int = 50, min_kl_improvement: float = 1e-7,
                 beta_tolerance: float = 1e-5, beta_search_iter: int = 50,
                 random_state: int | None = None, verbose: bool = False) -> None:
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        self.momentum = momentum
        self.final_momentum = final_momentum
        self.min_gain = min_gain
        self.patience = patience
        self.min_kl_improvement = min_kl_improvement
        self.beta_tolerance = beta_tolerance
        self.beta_search_iter = beta_search_iter
        self.random_state = random_state
        self.verbose = verbose

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X phải là ma trận 2 chiều có ít nhất 2 mẫu.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X không được chứa NaN hoặc vô cực.")
        if self.perplexity <= 0:
            raise ValueError("perplexity phải lớn hơn 0.")
        # Điều kiện chặt: tập dữ liệu cần lớn hơn ba lần perplexity.
        if 3.0 * self.perplexity >= X.shape[0]:
            maximum = X.shape[0] / 3.0
            raise ValueError(
                f"perplexity quá cao: cần 3 * perplexity < n_samples; "
                f"với {X.shape[0]} mẫu, perplexity phải nhỏ hơn {maximum:.3g}."
            )
        if self.n_components < 1 or self.n_iter < 1:
            raise ValueError("n_components và n_iter phải là số nguyên dương.")
        if self.learning_rate <= 0 or self.early_exaggeration <= 0:
            raise ValueError("learning_rate và early_exaggeration phải dương.")
        if self.min_gain <= 0:
            raise ValueError("min_gain phải lớn hơn 0.")
        if not isinstance(self.patience, (int, np.integer)) or self.patience < 1:
            raise ValueError("patience phải là số nguyên lớn hơn hoặc bằng 1.")
        if self.min_kl_improvement < 0:
            raise ValueError("min_kl_improvement phải lớn hơn hoặc bằng 0.")
        return X



    def fit(self, X: np.ndarray) -> "TSNE":
        """Học embedding, lưu vào ``embedding_`` rồi trả về chính đối tượng."""
        X = self._validate_input(X)
        D = squared_euclidean_distances(X)
        P_cond, self.betas_ = conditional_probabilities(
            D, self.perplexity, self.beta_tolerance, self.beta_search_iter)
        self.sigmas_ = np.sqrt(1.0 / (2.0 * self.betas_))
        self.P_ = symmetric_probabilities(P_cond)
        Y = initialize_embedding(len(X), self.n_components,
                                 np.random.default_rng(self.random_state))
        velocity = np.zeros_like(Y)
        gains = np.ones_like(Y)
        self.kl_history_ = []
        best_kl = np.inf
        rounds_without_improvement = 0
        updates_completed = 0
        self.stopped_early_ = False

        for iteration in range(self.n_iter):
            early = iteration < self.early_exaggeration_iter
            P_used = self.P_ * self.early_exaggeration if early else self.P_
            current_momentum = self.momentum if early else self.final_momentum
            Q, numerator = student_t_probabilities(Y)

            # Chỉ so sánh KL sau early exaggeration vì trước đó P đã được nhân
            # hệ số và không còn cùng thang đo với hàm mục tiêu KL(P || Q) gốc.
            # Dùng Q đã có ở đầu vòng lặp nên early stopping không tạo thêm một
            # phép tính khoảng cách O(N²).
            if not early:
                current_kl = kl_divergence(self.P_, Q)
                self.kl_history_.append((iteration, current_kl))
                improvement = best_kl - current_kl
                if improvement > self.min_kl_improvement:
                    best_kl = current_kl
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1

                if rounds_without_improvement >= self.patience:
                    self.stopped_early_ = True
                    if self.verbose:
                        print(
                            f"Dừng sớm tại vòng {iteration}: KL không cải thiện "
                            f"quá {self.min_kl_improvement:g} trong "
                            f"{self.patience} vòng liên tiếp."
                        )
                    break

            gradient = tsne_gradient(P_used, Q, Y, numerator)
            Y, velocity, gains = gradient_descent_step(
                Y, velocity, gradient, gains, self.learning_rate,
                current_momentum, self.min_gain)
            updates_completed += 1
            if self.verbose and (iteration % 100 == 0 or iteration == self.n_iter - 1):
                loss = kl_divergence(P_used, Q)
                print(f"Vòng lặp {iteration:4d}: KL divergence = {loss:.6f}")

        self.embedding_ = Y
        self.gains_ = gains
        final_Q, _ = student_t_probabilities(Y)
        self.kl_divergence_ = kl_divergence(self.P_, final_Q)
        self.best_kl_divergence_ = min(best_kl, self.kl_divergence_)
        self.n_iter_ = updates_completed
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Học mô hình và trả về embedding kích thước N x n_components."""
        return self.fit(X).embedding_

    def fit_tranform(self, X: np.ndarray) -> np.ndarray:
        """Alias cho cách viết ``fit_tranform`` trong đề bài (thiếu chữ s)."""
        return self.fit_transform(X)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = np.vstack((rng.normal(-3, 1, (50, 4)), rng.normal(3, 1, (50, 4))))
    model = TSNE(perplexity=20, n_iter=500, random_state=42, verbose=True)
    Y = model.fit_transform(X)
    print("Kích thước embedding:", Y.shape)
    print("KL divergence cuối:", model.kl_divergence_)
