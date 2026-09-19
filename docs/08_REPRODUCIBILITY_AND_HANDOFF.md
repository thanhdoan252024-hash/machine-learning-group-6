# Reproducibility and Handoff

## Rules
1. train-only scaler fit
2. train-only PCA fit
3. test transform only
4. labels excluded from PCA fit
5. no independent PCA refit for PCA90/PCA95
6. eigenvalues sorted descending
7. same official split

## Expected artifacts
- X_train_scaled.npy
- X_test_scaled.npy
- X_train_pca90.npy
- X_test_pca90.npy
- X_train_pca95.npy
- X_test_pca95.npy
- y_train.npy
- y_test.npy
- pca_from_scratch_parameters.npz
- pca_candidate_comparison.csv
- pca_final_summary.csv
- pca_reconstruction_analysis.csv

## Handoff acceptance
Người nhận xác nhận CP1–CP10 PASS và không có sklearn PCA/StandardScaler.

Floating-point có thể chênh rất nhỏ giữa BLAS/LAPACK implementations nhưng logic và thresholds phải tương đương.
