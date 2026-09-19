# 02 — t-SNE Theory Knowledge

## 1. Original-space similarities

For a sample `x_i`, t-SNE converts squared Euclidean distances to conditional Gaussian similarities:

`p(j|i) = exp(-β_i d_ij²) / Σ_{k≠i} exp(-β_i d_ik²)`, with `p(i|i)=0`.

Here:
- `d_ij² = ||x_i-x_j||²`;
- `β_i = 1/(2σ_i²)` is a precision parameter specific to sample `i`.

Using a different `β_i` for each row lets the method adapt to local density.

## 2. Entropy and perplexity

For one conditional row `P_i`:

`H(P_i) = -Σ_j p(j|i) log p(j|i)`

and

`Perplexity(P_i) = exp(H(P_i))`.

Perplexity can be interpreted approximately as an effective neighborhood size. The implementation searches for `β_i` so that the achieved entropy is close to `log(target_perplexity)`.

Higher beta means a narrower Gaussian, typically lower entropy and lower perplexity.

## 3. Symmetric high-dimensional distribution

After all conditional rows are computed:

`P = (P_cond + P_cond.T) / (2N)`.

The resulting matrix is symmetric, nonnegative, has zero diagonal and sums to one.

## 4. Low-dimensional distribution

In the embedding `Y`, t-SNE uses a heavy-tailed Student-t kernel:

`w_ij = (1 + ||y_i-y_j||²)^(-1)`

and

`q_ij = w_ij / Σ_{k≠l} w_kl`, with `q_ii=0`.

The heavy tail helps reduce the crowding problem by allowing moderately separated low-dimensional points to remain farther apart than with a Gaussian kernel.

## 5. Objective

The optimization minimizes:

`KL(P||Q) = Σ_ij p_ij log(p_ij/q_ij)`.

This direction of KL strongly penalizes cases where a high-probability neighbor in `P` receives very low probability in `Q`.

## 6. Exact gradient

The exact gradient is:

`∂C/∂y_i = 4 Σ_j (p_ij-q_ij)(y_i-y_j)/(1+||y_i-y_j||²)`.

The project verifies this gradient in three ways:
- vectorized implementation;
- brute-force reference implementation;
- central finite-difference numerical derivative on an independent tiny problem.

## 7. Optimization mechanisms

### Early exaggeration
During early iterations, `P` is multiplied by an exaggeration factor before computing the gradient. The original normalized `P` remains unchanged for true-KL monitoring.

### Momentum
Momentum combines the previous update with the current negative-gradient direction.

### Adaptive gains
Coordinate-wise gains increase when update/gradient signs change and decay otherwise, with a minimum gain floor.

### Re-centering
The embedding is centered after every update to remove irrelevant translation drift.

### Best-state restoration
The optimizer stores the actual embedding state corresponding to the best post-exaggeration true KL and restores that state at the end.

## 8. What t-SNE preserves

t-SNE is primarily a **local-neighborhood visualization method**. Therefore:
- local adjacency is meaningful;
- apparent cluster separation can be informative but is not a metric proof;
- global inter-cluster distances should not be interpreted literally;
- cluster area/size in 2D is not a faithful measure of original-space variance or population density.

## 9. Evaluation used in this project

### Trustworthiness
Penalizes points that become false neighbors in 2D.

### Continuity
Penalizes original-space neighbors that are lost in 2D.

Both are evaluated at `k={5,10,20,50}`.

### Seed stability
Two embeddings can differ by translation, reflection or rotation while representing similar neighborhoods. Therefore stability is measured by pairwise kNN-set overlap rather than raw coordinate MSE.

## 10. Complexity

This implementation is exact. Pairwise matrices are `N×N`, so memory is `O(N²)` and each optimization iteration is also dominated by `O(N²)` operations. For `N=5620`, one float64 `N×N` matrix is about 241 MiB.
