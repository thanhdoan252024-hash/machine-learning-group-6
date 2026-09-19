# Implementation Roadmap

## Phase 1 — Dataset Preparation
Input: UCI HAR ZIP/folder.  
Output: X/y train/test, feature_names, activity_map.  
Gate: CP1.

## Phase 2 — Manual Standardization
Fit mean/std trên train; transform train/test.  
Output: scaled train/test + scaler params.  
Gate: CP2.

## Phase 3 — PCA From Scratch
Covariance → `eigh` → sort → EVR → transform/inverse-transform.

## Phase 4 — Mathematical Validation
Symmetry, eigenvalues, orthogonality, EVR sum, score covariance.  
Gate: CP3.

## Phase 5 — Explained Variance
Scree plot, cumulative EVR, k80/k90/k95/k99.  
Gate: CP4.

## Phase 6 — Visualization
PC1/2, PC1/3, PC2/3, 3D.  
Gate: CP5.

## Phase 7 — Reconstruction
Train/test MSE theo k, monotonicity check.  
Gate: CP6.

## Phase 8 — Loading Analysis
PC1–PC5 top features.  
Gate: CP7.

## Phase 9 — Candidate Selection
PCA95 primary, PCA90 alternative.  
Gate: CP8.

## Phase 10 — Final PCA Datasets
Tạo PCA90/PCA95 bằng cùng eigenbasis.  
Gate: CP9.

## Phase 11 — Export
.npy/.npz/.csv/README.  
Gate: CP10.

# PCA END
Sau Phase 11: classification, không xử lý PCA lại.
