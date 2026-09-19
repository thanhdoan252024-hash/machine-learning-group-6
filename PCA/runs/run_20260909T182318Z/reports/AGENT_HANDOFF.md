# PCA handoff

Phases 1–11 completed; CP1–CP10 PASS.

Use ../outputs/X_train_scaled.npy and X_test_scaled.npy for the baseline, X_*_pca90.npy / X_*_pca95.npy for reduced representations, and y_train.npy / y_test.npy for labels.

Persisted mean/scale and eigenbasis are in pca_from_scratch_parameters.npz. Load with allow_pickle=False.

Keep the official held-out test split. These final artifacts fit preprocessing on all training data. If performing cross-validation within train, refit preprocessing on each training fold to avoid validation leakage.

Independent consumer checks: ../verification.json. File integrity: ../artifact_manifest.json.
