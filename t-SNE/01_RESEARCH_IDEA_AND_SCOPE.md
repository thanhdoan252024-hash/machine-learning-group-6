# 01 — Research Idea and Scope

## Research question

Can an exact t-SNE implementation be built from first principles with NumPy, independently validated at the mathematical and numerical levels, and evaluated on UCI Optdigits using neighborhood preservation and stochastic stability rather than visual appearance alone?

## Dataset and task

- Dataset: UCI Optical Recognition of Handwritten Digits (Optdigits)
- Samples: 5,620
- Input dimensionality: 64
- Labels: digits 0–9
- Target representation: 2 dimensions

This is a **dimensionality-reduction / visualization case study**, not a classification benchmark. Labels are not used by the t-SNE fitting objective. They are used only for deterministic stratified screening and post-hoc interpretation/visualization.

## Main technical scope

The case study implements and validates:

1. exact squared Euclidean distances;
2. Gaussian conditional similarities in the original space;
3. entropy/perplexity computation;
4. per-sample beta search;
5. symmetric joint probability distribution `P`;
6. Student-t low-dimensional distribution `Q`;
7. `KL(P||Q)` objective;
8. exact analytic t-SNE gradient;
9. numerical finite-difference gradient verification;
10. early exaggeration, momentum, adaptive gains, recentering, best-state restoration and early stopping;
11. synthetic structural validation;
12. perplexity sensitivity;
13. multi-scale Trustworthiness and Continuity;
14. multi-seed neighborhood-overlap stability;
15. primary/alternative configuration selection;
16. final exact 5,620-sample run;
17. reproducibility/package audits.

## Deliberate exclusions

The following are outside the defined case-study scope:

- Barnes–Hut or FFT/FIt-SNE acceleration;
- GPU/autodiff implementations;
- supervised use of labels during fitting;
- a parametric neural-network mapping;
- a simple `transform(X_new)` API for unseen samples;
- classification accuracy as the primary t-SNE quality metric;
- claiming a globally optimal perplexity.

## Scientific position

The final selected primary configuration is **p=40 among the evaluated candidates**. The protocol does not claim p=40 is universally optimal. Local-neighborhood evidence is prioritized over apparent cluster separation and over cross-perplexity KL comparison.
