# PCA Theory — Knowledge Document

## Standardization
\[
z_{ij} = \frac{x_{ij}-\mu_j}{\sigma_j}
\]
Mean/std chỉ fit trên train.

## Covariance
\[
C = \frac{1}{n-1}X^T X
\]
Với UCI HAR: \(C \in R^{561 \times 561}\).

## Eigendecomposition
\[
Cv_i = \lambda_i v_i
\]
Covariance symmetric nên dùng `np.linalg.eigh`, rồi sort eigenvalues giảm dần.

## Explained Variance
\[
EVR_i = \frac{\lambda_i}{\sum_j\lambda_j}
\]
\[
CEV(k)=\sum_{i=1}^k EVR_i
\]

## Projection
\[
Z=XW_k
\]

## Reconstruction
\[
\hat X=ZW_k^T
\]
\[
MSE=mean((X-\hat X)^2)
\]

## Loadings
Mỗi cột eigenvector là loading của feature trên một PC.  
Dùng `abs(loading)` để xếp hạng đóng góp, giữ signed loading để diễn giải hướng.

## Invariants để validation
- \(C \approx C^T\)
- \(V^TV \approx I\)
- sum(EVR) ≈ 1
- score covariance gần diagonal
- diagonal(score covariance) ≈ eigenvalues
- reconstruction MSE không tăng khi k tăng

## Visualization
Overlap trong PC1-PC2 không chứng minh PCA thất bại; thông tin phân biệt có thể nằm ở PC cao hơn.
