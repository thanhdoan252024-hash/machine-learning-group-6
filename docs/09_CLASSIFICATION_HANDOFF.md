# Classification Handoff

## PCA đã kết thúc
Không tính lại:
- scaler;
- covariance;
- eigenvectors;
- eigenvalues;
- k.

## Ba representation
1. Original standardized — 561 features
2. PCA90
3. PCA95

## Downstream evaluation
- Accuracy
- Macro-F1
- confusion matrix
- per-class performance
- training/inference time
- dimensionality reduction trade-off

Chỉ sau classification mới kết luận PCA90 hay PCA95 có downstream utility tốt hơn.
