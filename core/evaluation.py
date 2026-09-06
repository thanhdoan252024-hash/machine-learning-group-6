"""Đánh giá tự động t-SNE trên nhiều perplexity; đo KL, runtime, Trustworthiness; vẽ và lưu kết quả."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .tsne_numpy import TSNE


def trustworthiness(X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
    """Tính Trustworthiness T(k) theo định nghĩa gốc (Venna & Kaski 2001).

    T(k) = 1 - 2/(n*k*(2n-3k-1)) * sum_{i=1}^n sum_{j in U_k(i)} (r(i,j) - k)
    trong đó U_k(i) là các điểm nằm trong k-NN của i trong không gian nhúng
    nhưng KHÔNG nằm trong k-NN của i trong không gian gốc.
    r(i,j) là thứ hạng của j trong k-NN gốc của i.
    """
    n = X.shape[0]
    if n <= k:
        return 1.0

    # Khoảng cách trong không gian gốc
    D_high = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    # Khoảng cách trong không gian nhúng
    D_low = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)

    # Thứ hạng trong không gian gốc (0 = chính nó)
    ranks_high = np.argsort(np.argsort(D_high, axis=1), axis=1)
    # Các chỉ số k-NN trong không gian nhúng (bỏ qua chính nó)
    knn_low = np.argsort(D_low, axis=1)[:, 1:k+1]
    # Các chỉ số k-NN trong không gian gốc
    knn_high = np.argsort(D_high, axis=1)[:, 1:k+1]

    # Tìm các điểm trong knn_low nhưng KHÔNG trong knn_high
    trust_sum = 0.0
    for i in range(n):
        low_set = set(knn_low[i])
        high_set = set(knn_high[i])
        extra = low_set - high_set
        for j in extra:
            r = ranks_high[i, j]
            trust_sum += (r - k)

    denom = n * k * (2 * n - 3 * k - 1)
    if denom <= 0:
        return 1.0
    return float(1.0 - (2.0 * trust_sum) / denom)


def run_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    perplexities: Sequence[float] = (5, 10, 20, 30, 40, 50),
    n_iter: int = 1000,
    random_state: int = 42,
    output_dir: str | Path = "outputs",
    verbose: bool = True,
) -> pd.DataFrame:
    """Chạy t-SNE cho từng perplexity, thu thập metric và vẽ/lưu kết quả.

    Trả về DataFrame chứa: perplexity, kl_divergence, runtime_sec, trustworthiness.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    results_dir = output_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    records = []
    embeddings = {}

    for perp in perplexities:
        if verbose:
            print(f"\n=== Perplexity = {perp} ===")
        model = TSNE(
            n_components=2,
            perplexity=perp,
            n_iter=n_iter,
            random_state=random_state,
            verbose=verbose,
        )
        start = time.perf_counter()
        Y = model.fit_transform(X)
        elapsed = time.perf_counter() - start

        kl = model.kl_divergence_
        trust = trustworthiness(X, Y, k=5)

        records.append({
            "perplexity": perp,
            "kl_divergence": kl,
            "runtime_sec": elapsed,
            "trustworthiness": trust,
        })
        embeddings[perp] = Y

        if verbose:
            print(f"  KL Divergence   : {kl:.6f}")
            print(f"  Runtime (s)     : {elapsed:.2f}")
            print(f"  Trustworthiness : {trust:.6f}")

    df = pd.DataFrame(records)
    csv_path = results_dir / "experiment_evaluation.csv"
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"\nĐã lưu bảng kết quả: {csv_path}")

    # ---- Vẽ 4 hình ----
    # 1. Lưới embedding theo perplexity
    n_perp = len(perplexities)
    n_cols = min(3, n_perp)
    n_rows = (n_perp + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    for idx, perp in enumerate(perplexities):
        ax = axes[idx // n_cols, idx % n_cols]
        Y = embeddings[perp]
        scatter = ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap="tab10", s=6, alpha=0.7, edgecolors="none")
        ax.set_title(f"Perplexity = {perp}")
        ax.set_xticks([])
        ax.set_yticks([])
    # Ẩn axes thừa
    for idx in range(n_perp, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label="Digit", shrink=0.8)
    fig.suptitle("t-SNE Embeddings across Perplexities", fontsize=14)
    fig.tight_layout()
    grid_path = figures_dir / "tsne_perplexity_grid.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Perplexity vs KL Divergence
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["perplexity"], df["kl_divergence"], marker="o", color="tab:blue")
    ax.set_xlabel("Perplexity")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Perplexity vs KL Divergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    kl_path = figures_dir / "perplexity_vs_kl.png"
    fig.savefig(kl_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Perplexity vs Runtime
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["perplexity"], df["runtime_sec"], marker="s", color="tab:orange")
    ax.set_xlabel("Perplexity")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Perplexity vs Runtime")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    rt_path = figures_dir / "perplexity_vs_runtime.png"
    fig.savefig(rt_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Perplexity vs Trustworthiness
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["perplexity"], df["trustworthiness"], marker="^", color="tab:green")
    ax.set_xlabel("Perplexity")
    ax.set_ylabel("Trustworthiness (k=5)")
    ax.set_title("Perplexity vs Trustworthiness")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    tw_path = figures_dir / "perplexity_vs_trustworthiness.png"
    fig.savefig(tw_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"Đã lưu 4 hình vào: {figures_dir}")

    return df


def load_optdigits_csv(csv_path: str | Path = "data/processed/optdigits.csv") -> tuple[np.ndarray, np.ndarray]:
    """Đọc file CSV đã xử lý (64 features + 1 label)."""
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values.astype(np.float64)
    y = df.iloc[:, -1].values.astype(int)
    return X, y


if __name__ == "__main__":
    # Demo nhanh khi chạy file trực tiếp
    X, y = load_optdigits_csv()
    # Chỉ lấy 1000 mẫu đầu để demo nhanh
    X_sub, y_sub = X[:1000], y[:1000]
    run_evaluation(X_sub, y_sub, perplexities=(10, 30, 50), n_iter=500, verbose=True)