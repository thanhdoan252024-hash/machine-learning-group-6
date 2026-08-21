# Machine Learning Group 6 — LightGBM from scratch

Repository xây dựng các mô hình LightGBM-style bằng Python/NumPy cho hai bài
toán regression và classification. Phần classification dự đoán hỏng máy đã có
pipeline đánh giá, báo cáo, trực quan hóa và xuất artifact hoàn chỉnh.

## Cấu trúc chính

```text
classification/
  data/raw/machine_fail.csv
  lightgbm_classification.py
  classification_metrics.py
  machine_failure_prediction.ipynb
  evaluation/
    adapter.py
    runner.py
    run_machine_failure_evaluation.py
    input_validation.py
    manual_metrics.py
    manual_roc_auc.py
    reports.py
    visualizations.py
    exporters.py
    tests/
    outputs/
      tables/
      figures/
      predictions/
    docs/
      Ke_hoach_danh_gia_truc_quan_LightGBM_Classification_khong_sklearn.md
      PROJECT_INTEGRATION_NOTES.md
      ai_prompting_log.md
regression/
.gitignore
requirements.txt
```

## Nguyên tắc của phần classification

- `classification/lightgbm_classification.py` là model binary do nhóm tự xây
  dựng; không gọi model có sẵn từ package LightGBM.
- `sklearn` chỉ được dùng cho `train_test_split` có stratification.
- Không dùng `sklearn.metrics`, metric của LightGBM hoặc `model.score()`.
- Confusion Matrix, Accuracy, Precision, Recall, F1, ROC và AUC đều được tính
  thủ công bằng Python/NumPy trong `classification/evaluation/`.
- Pandas chỉ tổ chức bảng/xuất CSV; Matplotlib chỉ tạo hình.
- Không dùng `numpy.trapz` hoặc `numpy.trapezoid` để tính AUC.

Dataset `machine_fail.csv` có 10.000 mẫu và target `Machine failure`. Pipeline
dùng sáu feature không gây target leakage, chia train/test 80/20 với
`random_state=42` và `stratify=y`. Lớp dương `1` là máy hỏng.

## Cài đặt

Yêu cầu Python 3.10 trở lên. Từ repository root:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Trên macOS/Linux, thay đường dẫn Python trong virtual environment bằng
`./venv/bin/python`.

## Chạy kiểm thử

```powershell
.\venv\Scripts\python.exe -m unittest discover -s classification/evaluation/tests -p "test_*.py" -v
```

Bộ test bao phủ validation, metric thủ công, ROC-AUC, report/exporter, output
lifecycle, adapter model thật và hợp đồng dataset/split. Sau khi gom package,
full discovery đạt 90/90 test.

## Tái tạo kết quả classification

Chạy toàn bộ preprocessing, train, predict, đánh giá và xuất artifact:

```powershell
.\venv\Scripts\python.exe -m classification.evaluation.run_machine_failure_evaluation
```

Có thể chạy notebook `classification/machine_failure_prediction.ipynb` từ repo
root hoặc trực tiếp trong thư mục `classification`; notebook tự tìm repo root và
không còn phụ thuộc đường dẫn tuyệt đối trên một máy cụ thể.

## Artifact được tạo và commit

`classification/evaluation/outputs/` chứa:

- `tables/metrics_summary.csv`
- `tables/classification_report.csv`
- `tables/confusion_matrix_counts.csv`
- `tables/confusion_matrix_normalized.csv`
- `tables/roc_points.csv`
- `predictions/predictions.csv`
- Sáu hình PNG: hai confusion matrix, overall metrics, per-class metrics,
  correct/incorrect pie và ROC curve.
- `evaluation_manifest.json` ghi chính xác các artifact thuộc pipeline.

Mỗi lần chạy thành công sẽ ghi đè bộ kết quả hiện tại và dọn artifact ROC cũ
theo manifest; pipeline không xóa file ngoài danh sách do chính nó sở hữu.

## Kết quả baseline

Kết quả trong `classification/evaluation/outputs/` dùng model 100 estimator, threshold
`0.5` và test split 2.000 mẫu:

| Metric | Giá trị |
|---|---:|
| Accuracy | 0,986000 |
| Precision — máy hỏng | 0,900000 |
| Recall — máy hỏng | 0,661765 |
| F1-score — máy hỏng | 0,762712 |
| ROC-AUC | 0,974211 |

Confusion matrix gồm TN=1.927, FP=5, FN=23 và TP=45. Do lớp máy hỏng chỉ chiếm
3,39% toàn dataset, không nên diễn giải Accuracy tách rời Precision, Recall, F1
và ROC-AUC của lớp dương.

API tích hợp hiện tại:

```python
from classification.evaluation.adapter import (
    evaluate_classification_outputs,
    evaluate_fitted_classifier,
)
from classification.evaluation.runner import run_classification_evaluation
```

Chi tiết các quyết định tích hợp nằm trong
`classification/evaluation/docs/PROJECT_INTEGRATION_NOTES.md`; lịch sử prompt và
kiểm tra AI nằm trong `classification/evaluation/docs/ai_prompting_log.md`.
