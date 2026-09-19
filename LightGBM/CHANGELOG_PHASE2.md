# CHANGELOG — Phase 2: Shared Tree Engine + Model Selection Baselines

## Hoàn thành

### 1. Unified Tree/Split/Histogram core

Đã bổ sung ba module dùng chung:

- `lightgbm_from_scratch/core/histogram.py`
- `lightgbm_from_scratch/core/split.py`
- `lightgbm_from_scratch/core/tree.py`

Cả `LightGBMRegression` và `LightGBMClassification` hiện xây histogram, tìm
split và phát triển cây leaf-wise qua shared core. Các wrapper API lịch sử vẫn
được giữ để không phá notebook/test cũ.

### 2. Early stopping và evaluation history

Cả hai estimator hỗ trợ:

- `eval_set=(X_val, y_val)`
- `early_stopping_rounds`
- `best_iteration_`
- `best_score_`
- `n_estimators_`
- `evals_result_`

Regression theo dõi RMSE; Classification theo dõi binary log-loss.

### 3. Regression case study chính thức

Scenario: **pre-exam student score prediction**.

- 100,000 mẫu.
- Split raw rows 70/15/15 trước preprocessing.
- Preprocessor chỉ fit trên train.
- Loại ID, target-derived và post-exam columns, bao gồm
  `questions_attempted` và `questions_correct`.
- 37 feature hợp lệ cho primary scenario.

Kết quả test baseline Phase 2:

- MAE: `7.5452`
- RMSE: `9.4816`
- R²: `0.6360`

Best iteration chạm budget `80/80`, vì vậy đây là baseline cho hyperparameter
study chứ chưa phải final configuration.

### 4. Classification case study chính thức

Scenario: **predictive maintenance / machine failure**.

- 10,000 mẫu, positive ratio 3.39%.
- Stratified split 70/15/15.
- Early stopping chọn 134 boosting iterations.
- Threshold chỉ tune trên validation bằng F1.
- Threshold chọn: `0.37`.

Test tại threshold 0.37:

- Accuracy: `0.9767`
- Precision: `0.6905`
- Recall: `0.5686`
- F1: `0.6237`
- Balanced Accuracy: `0.7798`
- ROC-AUC: `0.9629`
- PR-AUC / Average Precision: `0.7349`

So với threshold 0.5, Recall tăng từ `0.4510` lên `0.5686`, đổi lại Precision
giảm từ `1.0000` xuống `0.6905`. Đây là trade-off cần đánh giá theo chi phí
false negative trong predictive maintenance.

### 5. Artifact reproducibility

Regression export:

- `metrics.json`
- `predictions.csv`
- `feature_importance.csv`
- `learning_history.csv`
- 5 report-ready figures

Classification export:

- `metrics.json`
- `predictions.csv`
- `feature_importance.csv`
- `learning_history.csv`
- `threshold_analysis.csv`
- `roc_points_test.csv`
- `pr_points_test.csv`
- 7 report-ready figures

### 6. Tests / correctness

- Phase 1: 162 pass + 2 subtests.
- Phase 2: **168 pass + 2 subtests, 0 fail**.
- Core mathematical validation: all checks pass.

## Chưa gọi là final model

Phase 2 tạo một baseline khoa học, leakage-safe và validation-driven. Các metric
trên chưa được dùng để tuyên bố cấu hình tối ưu. Phase 3 sẽ thực hiện controlled
hyperparameter study, ablation GOSS/EFB/regularization/feature sampling và so
sánh reference models trước khi chọn final configuration.
