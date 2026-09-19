# FINAL PCA REPORT TEMPLATE

## 1. Problem Statement

Mô tả mục tiêu giảm chiều UCI HAR bằng PCA from scratch.

## 2. Dataset

- samples
- features
- classes
- sensors
- split

## 3. Constraints

Nêu rõ không dùng sklearn PCA/StandardScaler.

## 4. Method

### 4.1 Standardization
### 4.2 Covariance Matrix
### 4.3 Eigendecomposition
### 4.4 Explained Variance
### 4.5 Projection
### 4.6 Reconstruction
### 4.7 Loading Analysis

## 5. Validation

Tóm tắt CP1–CP10.

## 6. Results

### 6.1 Scree Plot
### 6.2 Cumulative Variance
### 6.3 k80/k90/k95/k99
### 6.4 2D/3D Visualization
### 6.5 Reconstruction Error
### 6.6 Loadings

## 7. Final PCA Candidates

- PCA90
- PCA95

## 8. Interpretation

PCA giảm chiều được bao nhiêu và giữ lại bao nhiêu variance?

## 9. Limitations

- PCA là linear method.
- 2D/3D visualization chỉ phản ánh vài PC đầu.
- Explained variance không trực tiếp tương đương classification performance.

## 10. Downstream Handoff

So sánh:
- Original standardized
- PCA90
- PCA95

## 11. Reproducibility

Mô tả train-only fit, artifact export và environment record.
