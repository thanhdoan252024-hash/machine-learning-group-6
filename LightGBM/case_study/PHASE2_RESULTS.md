# Phase 2 — Official Leakage-Safe Baselines

## Regression

Primary scenario: dự đoán `exam_score` trước khi kỳ thi hoàn tất.

| Split | MAE | RMSE | R² |
|---|---:|---:|---:|
| Train | 7.6178 | 9.5202 | 0.6410 |
| Validation | 7.6380 | 9.5420 | 0.6394 |
| Test | **7.5452** | **9.4816** | **0.6360** |

Train/validation/test khá sát nhau. Việc R² giảm mạnh so với package ban đầu là
kết quả phù hợp với việc loại các feature post-exam/target-derived, đặc biệt
`questions_correct` vốn có tương quan tuyệt đối khoảng 0.975 với `exam_score`.
Baseline Phase 2 vì vậy thực tế hơn cho prediction scenario đã định nghĩa.

Top gain-importance hiện tại: `previous_exam_score`, `exam_difficulty`,
`study_hours_per_day`, `exam_preparation_days`, `practice_tests_completed`.

## Classification

Dataset có 339/10,000 positive samples (3.39%), do đó report dùng thêm Recall,
Balanced Accuracy và PR-AUC thay vì chỉ Accuracy/ROC-AUC.

Threshold 0.37 được chọn **chỉ trên validation** bằng F1.

| Split | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Train | 0.8768 | 0.7806 | 0.8259 | 0.9931 | 0.9110 |
| Validation | 0.7174 | 0.6471 | 0.6804 | 0.9770 | 0.7507 |
| Test | **0.6905** | **0.5686** | **0.6237** | **0.9629** | **0.7349** |

Test confusion matrix tại threshold 0.37:

- TN = 1436
- FP = 13
- FN = 22
- TP = 29

Threshold 0.5 trên cùng test split cho Precision 1.0 nhưng Recall chỉ 0.4510.
Threshold tuning đã tăng Recall nhưng vẫn còn 22/51 failure bị bỏ sót, nên Phase
3 cần đánh giá imbalance-oriented configuration thay vì tuyên bố baseline này
là tối ưu.

Top gain-importance: `Torque [Nm]`, `Rotational speed [rpm]`,
`Tool wear [min]`, `Air temperature [K]`, `Process temperature [K]`, `Type`.

## Kết luận Phase 2

Milestone này hoàn thành mục tiêu tạo **hai baseline chính thức có validation,
early stopping, artifact truy vết và không dùng test cho model selection**.
Bước kế tiếp là controlled hyperparameter study + ablation + reference models.
