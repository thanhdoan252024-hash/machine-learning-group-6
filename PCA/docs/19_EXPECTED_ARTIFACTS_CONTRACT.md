# Expected Artifacts Contract

## Required after PCA completion

### Data
```text
X_train_scaled.npy
X_test_scaled.npy
X_train_pca90.npy
X_test_pca90.npy
X_train_pca95.npy
X_test_pca95.npy
y_train.npy
y_test.npy
```

### Parameters
```text
pca_from_scratch_parameters.npz
```

Required keys:
- train_mean
- train_scale
- zero_variance_mask
- eigenvalues
- eigenvectors
- explained_variance_ratio
- cumulative_variance
- k90
- k95

### Tables
```text
pca_candidate_comparison.csv
pca_final_summary.csv
pca_reconstruction_analysis.csv
```

### Documentation
```text
README.txt
PHASE_REPORTS
RESULTS
ISSUE_LOG
DECISION_LOG
ENVIRONMENT_RECORD
FINAL_PCA_REPORT
```

## Validation contract

Consumer environment must verify:
- arrays are finite;
- row counts match labels;
- PCA90 columns = k90;
- PCA95 columns = k95;
- PCA90 equals PCA95 prefix within tolerance;
- parameter file contains required keys.
