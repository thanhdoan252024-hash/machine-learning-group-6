# Final PCA report

Objective: implement PCA with NumPy on UCI HAR and quantify dimensionality reduction and reconstruction loss.

## Data and method

The official train/test split is retained. Standardization uses train mean and population standard deviation.
The centered train covariance uses n−1. NumPy eigh yields the orthonormal eigenbasis, sorted by decreasing eigenvalue.
Train-fitted eigenvectors transform both train and test. Labels are used for audit and visualization only.

## Results

| Dataset representation | Dimensions | Retained variance | Reduction ratio |
| --- | --- | --- | --- |
| Original standardized | 561 | 1 | 0 |
| PCA90 | 63 | 0.90053451 | 0.88770053 |
| PCA95 | 102 | 0.95075582 | 0.81818182 |

| Configuration | Target variance | k | Actual retained variance | Dimensions reduced | Reduction ratio | Train reconstruction MSE | Test reconstruction MSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PCA80 | 0.8 | 26 | 0.80180504 | 535 | 0.95365419 | 0.19819496 | 0.20112439 |
| PCA90 | 0.9 | 63 | 0.90053451 | 498 | 0.88770053 | 0.099465493 | 0.1082636 |
| PCA95 | 0.95 | 102 | 0.95075582 | 459 | 0.81818182 | 0.049244178 | 0.055832152 |
| PCA99 | 0.99 | 179 | 0.99019268 | 382 | 0.68092692 | 0.0098073208 | 0.012421731 |

## Validation and interpretation

CP1–CP10 and the transform/inverse-transform functional test passed. All 13 figures are saved under ../figures/.
PCA90 is the compression-oriented candidate; PCA95 is the primary variance-preserving candidate.
Explained variance and training reconstruction error are related views of the same approximation objective.
Low-dimensional scatter overlap does not, by itself, establish classification performance.

## Scope and limitations

Variance retained is not classification accuracy. Classification and runtime comparisons have not been performed in this run.
The PCA class assumes input centered by the preceding scaler. Inverse transform reconstructs standardized features.
Eigenvector signs and tiny numerical residuals can differ across BLAS/LAPACK implementations.

## Reproducibility

The executed notebook, HTML, input hashes, environment, locked package versions and output hashes accompany this report.
See ../README.md for rerun and independent artifact verification commands.
