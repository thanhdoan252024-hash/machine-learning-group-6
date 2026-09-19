# Results and Evidence Guide

Tài liệu này quy định **mỗi phase phải ghi lại bằng chứng gì** để một môi trường khác có thể kiểm tra lại kết quả.

## Nguyên tắc

Không chỉ ghi `PASS`. Mỗi checkpoint phải có:

1. Input đã dùng.
2. Code/Prompt ID.
3. Metric kiểm tra.
4. Giá trị thực tế.
5. Tolerance/điều kiện pass.
6. Kết luận PASS/FAIL.
7. File/figure liên quan.
8. Ghi chú nếu có sai khác numerical.

## Evidence tối thiểu theo checkpoint

| Checkpoint | Evidence cần lưu |
|---|---|
| CP1 | shapes, NaN/Inf count, class counts, number of features |
| CP2 | max abs mean, max abs std error, zero-variance count |
| CP3 | symmetry error, min eigenvalue, orthogonality error, EVR sum, off-diagonal score covariance |
| CP4 | k80/k90/k95/k99, cumulative EV values, scree/cumulative plots |
| CP5 | PC1/2, PC1/3, PC2/3, 3D plots |
| CP6 | reconstruction table + curve + full-PC MSE |
| CP7 | top loadings PC1–PC5 + eigenvector norm |
| CP8 | candidate comparison PCA80/90/95/99 + decision rationale |
| CP9 | final shapes + nested-basis error |
| CP10 | exported file list + existence check |

## Quy tắc đặt tên evidence

Khuyến nghị:

```text
evidence/
├── CP01_dataset_audit.txt
├── CP02_standardization.txt
├── CP03_pca_math_validation.txt
├── CP04_variance_summary.csv
├── CP04_scree_plot.png
├── CP04_cumulative_variance.png
├── CP05_pc1_pc2.png
├── CP05_pc1_pc3.png
├── CP05_pc2_pc3.png
├── CP05_pc1_pc2_pc3_3d.png
├── CP06_reconstruction.csv
├── CP06_reconstruction_curve.png
├── CP07_loadings_pc1.csv
├── ...
├── CP08_candidate_comparison.csv
├── CP09_final_dataset_validation.txt
└── CP10_export_manifest.txt
```

## Acceptance rule

Nếu checkpoint không có evidence đủ để người khác kiểm tra, checkpoint đó chưa được xem là hoàn tất về mặt handoff.

## B?ng ch?ng ?? th?c hi?n

Runner `scripts/run_pca.py` t? l?u c?c file trong `runs/run_<UTC>/`: notebook th?c thi v? HTML ? th? m?c run; PNG trong `figures/`; b?ng/arrays trong `outputs/`; metric v? log t?ng phase trong `evidence/`; b?o c?o trong `reports/`. T?n log l? `PHASE_01.txt` ??n `PHASE_11.txt`, li?n k?t v?i CP v? tolerance trong `evidence/checkpoints.json`. Xem [b? k?t qu? hi?n c?](../runs/README.md).
