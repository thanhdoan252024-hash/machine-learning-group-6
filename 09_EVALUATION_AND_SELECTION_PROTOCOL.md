# 09 — Evaluation and Selection Protocol

## P09 — Controlled perplexity sensitivity

Evaluate perplexities:

`{5,10,20,30,40,50}`

on the same deterministic 1,000-sample stratified subset. Hold optimizer settings and random seed fixed so the main changing factor is perplexity.

Record:
- embedding;
- runtime;
- initial/best/restored KL;
- iterations;
- achieved perplexity diagnostics.

KL is an optimization diagnostic. Because each perplexity creates a different high-dimensional `P`, cross-perplexity final KL is not treated as a common leaderboard score.

## P10 — Multi-scale neighborhood preservation

For each stored embedding compute:
- Trustworthiness `T@5, T@10, T@20, T@50`;
- Continuity `C@5, C@10, C@20, C@50`.

An identity-geometry test must give `T=C=1` within floating-point tolerance.

Create a Pareto shortlist over the eight neighborhood metrics. If needed, use the conservative worst-neighborhood score only to reduce the shortlist to a manageable number of candidates.

## P11 — Multi-seed robustness

For each shortlisted perplexity, run seeds:

`{0,42,123}`.

For every seed compute the same multi-scale T/C metrics. Compare embeddings structurally via pairwise kNN-set overlap at `k={5,10,20,50}`.

Do not use raw coordinate MSE because equivalent t-SNE structures may differ by rotation, reflection or translation.

## Final selection rule

Selection is lexicographic/robust rather than a weighted score:

1. robust worst neighborhood preservation;
2. worst seed stability;
3. mean neighborhood preservation;
4. mean seed stability;
5. runtime only as a final tie-break.

Verified reference decision:
- PRIMARY = `40`
- ALTERNATIVE = `30`

This is the best decision among the evaluated candidates under the declared protocol, not a claim of global optimality.
