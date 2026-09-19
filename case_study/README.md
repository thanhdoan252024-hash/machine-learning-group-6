# LightGBM From Scratch — Case Study Workspace

Thư mục này là workspace tái lập cho case study Regression + Classification.
Mục tiêu là mọi con số trong báo cáo cuối đều truy ngược được về script,
config và artifact tương ứng.

## Trạng thái

- [x] Freeze package gốc và baseline tests.
- [x] Tạo shared core cho objective, binning, GOSS, regularization và EFB.
- [x] Regression/Classification dùng shared primitives nhưng giữ API cũ.
- [x] Sửa EFB theo hướng histogram-only, không đổi semantics của feature split.
- [x] Tối ưu Regression để bin dữ liệu một lần thay vì mỗi tree.
- [x] Bổ sung correctness tests cho gradient finite difference và EFB equivalence.
- [x] Chuẩn hóa tree/split/histogram engine dùng chung.
- [x] Regression train/validation/test leakage-safe experiment (100,000 rows).
- [x] Classification validation + imbalance/threshold experiment.
- [x] Early stopping + evaluation history cho cả hai estimator.
- [x] Hyperparameter study.
- [x] Ablation study.
- [x] Reference-model comparison.
- [x] Final case-study report theo mẫu PCA (Phase 4).

## Nguyên tắc thực nghiệm

1. Split dữ liệu trước mọi preprocessing học từ dữ liệu.
2. Validation dùng cho model selection / threshold selection; test chỉ dùng báo cáo cuối.
3. Mọi experiment lưu config, metrics, predictions và figures.
4. Correctness validation phải pass trước khi dùng kết quả model trong báo cáo.
5. EFB chỉ được coi là hoạt động khi histogram tái tạo đúng histogram feature gốc.

## Phase 2 outputs

- `artifacts/regression/`: official leakage-safe regression baseline + figures.
- `artifacts/classification/`: validation-driven classification baseline + figures.
- `artifacts/phase2_validation.json`: machine-readable milestone summary.
- `PHASE2_RESULTS.md`: diễn giải kết quả chính và giới hạn hiện tại.

## Phase 3 outputs

- `artifacts/phase3/regression_hyperparameter_study.csv`: controlled regression screening.
- `artifacts/phase3/classification_hyperparameter_study.csv`: controlled classification screening.
- `artifacts/phase3/*_ablation.csv`: component ablation evidence.
- `artifacts/phase3/*_reference_comparison.csv`: comparison to reference models.
- `artifacts/phase3/efb_sparse_microbenchmark.json`: sparse EFB correctness/performance evidence.
- `artifacts/phase3/figures/`: report-ready Phase 3 figures.
- `artifacts/phase3/phase3_summary.json`: consolidated machine-readable summary.
- `PHASE3_RESULTS.md`: interpretation and final model freeze.

## Phase 4 output

- `report/Case_Study_Thuc_nghiem_LightGBM_Regression_Classification.docx`: báo cáo Word final, đã render và kiểm tra layout toàn bộ 19 trang.

## Final submission clarifications

The root `README.md` and final Word report now explicitly document:

- library boundaries (what is allowed in the from-scratch implementation vs reference only);
- Regression screening on a fixed 5,000-train + 2,000-validation subset followed by full 70k/15k retraining/early stopping;
- native missing-bin capability vs the train-only mean/mode imputation used for final 100k Regression metrics;
- Official LightGBM reference protocol (not strict iso-iteration; actual best iterations 151/230 remain below scratch caps 240/450);
- Phase-1 80/20 classification moved to `legacy/phase1_classification/` so the final 70/15/15 pipeline is visually unambiguous.

The final report renders cleanly as **20 pages** after these clarifications.
