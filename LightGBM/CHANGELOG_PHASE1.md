# Phase 1 Changelog — Correctness Foundation

## Completed

1. Frozen the uploaded package as `artifacts/baseline/machine-learning-group-6-LightGBM-original.zip`.
2. Verified the original test baseline: 55 regression tests and 97 classification tests (+2 subtests).
3. Added `lightgbm_from_scratch/` shared core modules for:
   - quantile binning,
   - regression/binary objectives,
   - GOSS,
   - L1/L2 leaf optimization and split gain,
   - correctness-first EFB histogram construction.
4. Rewired both public estimators to use shared primitives while preserving their historical import paths.
5. Fixed EFB design:
   - sparsity is detected on raw values, not post-binning ids;
   - tree splits remain on original features;
   - EFB is used only to accelerate histogram construction;
   - original-feature histograms are reconstructed exactly.
6. Optimized `LightGBMRegression` so training/prediction bins a matrix once rather than once per tree.
7. Added mathematical validation and dedicated core tests, including binary finite-difference gradient checking and EFB-vs-brute-force histogram equivalence.
8. Added a raw regression data audit and a leakage-safe pre-exam train/validation/test pipeline.
9. Regenerated classification artifacts from source so repository outputs match executable code.

## Verification

Current suite after Phase 1: 161 tests passed (+2 subtests).

Core mathematical validation:
- regression gradient max abs error: 0.0
- binary finite-difference gradient max abs error: 2.8178104294340756e-10
- EFB histogram equivalence max abs error: 4.440892098500626e-16

## Next milestone

Unify the remaining tree/split engine, then add validation/early-stopping and the final Regression/Classification experiments used by the case-study report.
