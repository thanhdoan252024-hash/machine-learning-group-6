# Execution Checklist

?? ho?n th?nh trong c?c l?n ch?y ???c l?u t?i [runs/README.md](../runs/README.md). Metric, tolerance, log, bi?u ?? v? ki?m tra artifacts n?m trong t?ng b? k?t qu?; xem b? m?i nh?t trong `runs/latest.json`.

## Before starting

- [x] Đã đọc README
- [x] Đã đọc Research Idea
- [x] Đã đọc Theory
- [x] Đã đọc Roadmap
- [x] Đã xác nhận dataset chính xác
- [x] Đã ghi environment record

## Phase 1
- [x] Load dataset
- [x] Audit shape
- [x] Check NaN/Inf
- [x] Check labels/classes
- [x] CP1 evidence saved
- [x] CP1 PASS

## Phase 2
- [x] Fit scaler on train only
- [x] Transform test with train stats
- [x] CP2 evidence saved
- [x] CP2 PASS

## Phase 3–4
- [x] Covariance built manually
- [x] `np.linalg.eigh`
- [x] Sort descending
- [x] Transform/inverse-transform
- [x] Mathematical validation
- [x] CP3 evidence saved
- [x] CP3 PASS

## Phase 5
- [x] Scree plot
- [x] Cumulative plot
- [x] k80/k90/k95/k99
- [x] CP4 PASS

## Phase 6
- [x] PC1/2
- [x] PC1/3
- [x] PC2/3
- [x] 3D plot
- [x] CP5 PASS

## Phase 7
- [x] Reconstruction table
- [x] Reconstruction curve
- [x] Monotonic check
- [x] CP6 PASS

## Phase 8
- [x] PC1–PC5 loadings
- [x] Norm check
- [x] CP7 PASS

## Phase 9
- [x] Candidate comparison
- [x] PCA95 primary
- [x] PCA90 alternative
- [x] CP8 PASS

## Phase 10
- [x] Build PCA90/PCA95
- [x] Nested basis check
- [x] CP9 PASS

## Phase 11
- [x] Export arrays
- [x] Export parameters
- [x] Export CSV tables
- [x] Export README
- [x] CP10 PASS

## End
- [x] Final report completed
- [x] Agent handoff completed
- [x] Issue log reviewed
- [x] Decision log reviewed
- [x] PCA marked DONE
- [x] Ready for classification
