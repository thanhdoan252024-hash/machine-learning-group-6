# Changelog — Phase 3

## Added
- Controlled validation-only hyperparameter screening for Regression and Classification.
- GOSS, EFB, regularization, feature-fraction and positive-weight ablation controls.
- `scale_pos_weight` support for binary classification.
- Vectorized tree prediction to remove row-by-row Python traversal bottleneck.
- EFB sparse microbenchmark with equivalence and runtime measurements.
- Pre-specified reference comparisons against Official LightGBM, Random Forest, Logistic Regression and a Dummy regression baseline.
- Phase 3 summary JSON, result tables and report-ready visualization figures.

## Fixed
- Shared histogram allocation now preserves the dedicated missing-value bin even when observed bins do not occupy the complete range.
- Official-LightGBM classification benchmark now passes NumPy arrays because the source feature names contain characters rejected by LightGBM's feature-name validator.
- Reference benchmark labels use the installed LightGBM package version dynamically.

## Final selected models
- Regression: 125 effective boosting rounds; test RMSE 9.2096; test R² 0.6566.
- Classification: 195 effective boosting rounds; validation-selected threshold 0.22; test PR-AUC 0.7825; Recall 0.7451; F1 0.6726.

## Interpretation
EFB does not activate on the two dense main datasets, but the sparse microbenchmark compresses 40 mutually-exclusive features into one bundle with approximately 4.27x histogram speedup and numerical equivalence. GOSS contributes measurable validation gains in both tasks.
