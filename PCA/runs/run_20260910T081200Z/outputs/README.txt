UCI HAR PCA FROM SCRATCH - EXPORTED ARTIFACTS
==============================================

Original features: 561

Primary PCA candidate:
- PCA95
- k = 102
- retained variance = 0.95075582

Alternative PCA candidate:
- PCA90
- k = 63
- retained variance = 0.90053451

IMPORTANT:
1. Standardization statistics were fitted ONLY on X_train.
2. PCA eigenvectors/eigenvalues were fitted ONLY on X_train_scaled.
3. X_test was only transformed using training-fitted parameters.
4. No sklearn StandardScaler or sklearn PCA was used.
5. For classification compare Original, PCA90, and PCA95.

REPRODUCIBILITY EVIDENCE
Evaluation tables and feature/activity metadata are in this outputs directory.
13 PNG figures are in ../figures; numerical checks are in ../evidence/run_metrics.json.
feature_index and Feature index are zero-based; uci_feature_index is one-based.
Reconstruction MSE is measured in standardized feature space.
NPY/NPZ files were reloaded without pickle and compared exactly to computed arrays.
Raw train/test projections using saved parameters passed atol=1e-10, rtol=0.
