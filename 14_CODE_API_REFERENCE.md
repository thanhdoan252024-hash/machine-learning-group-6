# 14 — Code/API Reference

This document summarizes the public functions/classes used by the case study. The authoritative implementation is under `src/`.

## `src/tsne_from_scratch.py`

### `squared_euclidean_distances(X)`
Computes the exact pairwise squared Euclidean matrix using the Gram-matrix identity. The function validates shape and finite values, symmetrizes numerical noise, clips tiny negative round-off values, and forces a zero diagonal.

### `entropy_and_probabilities(distance_row, beta, self_index)`
Builds one Gaussian conditional-distribution row `p(j|i)` for a fixed precision `beta`. The self probability is zero. Entropy uses natural logarithms and perplexity is `exp(H)`.

### `binary_search_beta(...)`
Searches a positive `beta_i` so that the row entropy approaches `log(target_perplexity)`. Adaptive lower/upper bounds are used; no SciPy optimizer is required.

### `conditional_probability_matrix(D, target_perplexity, ...)`
Runs the beta search independently for every sample and returns conditional probabilities, betas, achieved perplexities, entropies, iteration counts, and convergence flags.

### `symmetrize_probabilities(P_cond)`
Constructs the joint high-dimensional distribution:

`P = (P_cond + P_cond.T) / (2N)`.

### `initialize_embedding(...)`
Creates deterministic Gaussian initialization from a NumPy random generator.

### `low_dimensional_affinities(Y)`
Computes Student-t affinities in the low-dimensional embedding:

`q_ij ∝ (1 + ||y_i-y_j||²)^(-1)`.

### `kl_divergence(P, Q)`
Computes `KL(P || Q)` on entries where `P > 0`.

### `tsne_gradient(P, Q, numerator, Y)`
Vectorized exact t-SNE gradient:

`4 Σ_j (p_ij-q_ij)(y_i-y_j)/(1+||y_i-y_j||²)`.

### `optimize_tsne_embedding(...)`
Exact iterative optimizer with:
- early exaggeration;
- momentum schedule;
- adaptive gains;
- embedding recentering;
- true-KL monitoring;
- best-state restoration;
- early stopping;
- KL and gradient histories.

### `TSNEFromScratch`
Educational fit/fit_transform wrapper. It intentionally does **not** provide an out-of-sample `transform` method because standard non-parametric t-SNE does not learn a simple reusable projection mapping.

## `src/evaluation.py`

### `knn_indices_from_distances(D, k)`
Returns top-k non-self neighbors from a pairwise distance matrix.

### `neighbor_order_and_ranks(D)`
Builds complete neighbor ordering and 1-based rank matrix.

### `trustworthiness_from_ranks(...)`
Measures false-neighbor introduction after embedding.

### `continuity_from_ranks(...)`
Measures original-neighbor loss after embedding.

### `neighborhood_quality(...)`
Computes Trustworthiness and Continuity at multiple neighborhood scales.

### `neighborhood_overlap_score(a, b)`
Structural stability metric between two embeddings based on kNN set overlap; invariant to rotation/reflection/translation of the coordinate system.

### `pareto_frontier(df, metric_columns)`
Returns non-dominated configurations when every supplied metric is maximized.

## Scripts

- `scripts/verify_environment.py`: environment/package preflight.
- `scripts/run_all_checks.py`: pytest + static audit + empirical reference audit + smoke test.
- `scripts/run_case_study.py`: command-line reproduction for `quick`, `screening`, or `full` profiles.
- `scripts/run_notebook.py`: clean-kernel notebook execution with profile control.
- `scripts/audit_rerun_outputs.py`: protocol-aware audit for outputs reproduced on another machine.
- `scripts/audit_empirical_outputs.py`: strict audit of bundled verified reference outputs.
