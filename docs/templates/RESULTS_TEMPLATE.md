# PCA RESULTS TEMPLATE

## Dataset Summary

- Train samples:
- Test samples:
- Original features:
- Classes:
- Missing values:
- Infinite values:

## Standardization Results

- Max absolute train mean:
- Max absolute std error:
- Near-zero variance features:
- CP2: PASS / FAIL

## PCA Mathematical Validation

- Covariance symmetry error:
- Minimum eigenvalue:
- Orthogonality error:
- EVR sum:
- Max score-covariance off-diagonal:
- Diagonal-vs-eigenvalue error:
- CP3: PASS / FAIL

## Explained Variance

| Threshold | k | Actual variance | Reduction ratio |
|---:|---:|---:|---:|
| 80% | | | |
| 90% | | | |
| 95% | | | |
| 99% | | | |

## Reconstruction

| k | Train MSE | Test MSE | Retained variance |
|---:|---:|---:|---:|
| | | | |

## Loading Summary

### PC1
1.
2.
3.

### PC2
1.
2.
3.

### PC3
1.
2.
3.

## Final Candidate Comparison

| Configuration | k | Retained variance | Reduction ratio | Train MSE | Test MSE |
|---|---:|---:|---:|---:|---:|
| PCA80 | | | | | |
| PCA90 | | | | | |
| PCA95 | | | | | |
| PCA99 | | | | | |

## Final Decision

- Primary candidate:
- Alternative candidate:
- Why:
- What remains to verify in classification:
