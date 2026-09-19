# 12 — Results and Evidence Guide

Report evidence in the following order so claims follow from progressively stronger validation.

## 1. Data integrity

Show:
- 5,620 samples;
- 64 features;
- no NaN/Inf;
- no accidental index feature;
- class distribution and feature-variance audit.

## 2. Mathematical component validation

Show:
- vectorized distance vs brute-force distance;
- conditional-probability normalization;
- monotonic beta/perplexity behavior;
- perplexity-search convergence;
- P and Q invariants;
- KL identity/translation checks.

## 3. Gradient correctness

Report analytic-vs-brute-force and finite-difference evidence. The final reference finite-difference relative error is approximately `1.45e-09`.

## 4. Optimizer behavior

Show:
- finite non-collapsed embedding;
- true KL improvement;
- best-state restoration;
- convergence history.

## 5. Synthetic validation

Use neighborhood overlap relative to a random baseline; labels are post-hoc only.

## 6. P09 sensitivity

Present the six perplexity runs. Do not declare a winner based solely on figure appearance or cross-perplexity KL.

## 7. P10 neighborhood preservation

Present T/C at all four neighborhood scales. Use Pareto evidence to motivate the shortlist.

## 8. P11 stochastic robustness

Present per-seed quality, pairwise structural stability, aggregate mean/worst results, and the primary/alternative decision.

## 9. P12 final run

Present:
- full input/output shape;
- selected perplexity;
- initial/final KL;
- iteration count;
- runtime;
- final embedding figure;
- **direct full-data Trustworthiness/Continuity at k={5,10,20,50};**
- KL and gradient histories.

The final full-data quality table is `outputs/tables/final_neighborhood_quality_p<PRIMARY>.csv`. This closes the evaluation loop by measuring the exact final embedding rather than relying only on the screening subset.

## Interpretation rules

Do not claim:
- visual separation alone proves quality;
- a lower KL from a different perplexity automatically means a better embedding;
- 2D cluster area equals original-space variance/density;
- inter-cluster 2D distance is a faithful global metric;
- p=40 is globally optimal.
