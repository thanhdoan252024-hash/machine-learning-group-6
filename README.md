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
      evaluation_manifest.json
      train/
        evaluation_manifest.json
        tables/
        figures/
        predictions/
      test/
        evaluation_manifest.json
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
lifecycle, adapter model thật và hợp đồng dataset/split. Full discovery hiện đạt
97/97 test.

## Tái tạo kết quả classification

Chạy toàn bộ preprocessing, train, predict, đánh giá và xuất artifact:

```powershell
.\venv\Scripts\python.exe -m classification.evaluation.run_machine_failure_evaluation
```

Có thể chạy notebook `classification/machine_failure_prediction.ipynb` từ repo
root hoặc trực tiếp trong thư mục `classification`; notebook tự tìm repo root và
không còn phụ thuộc đường dẫn tuyệt đối trên một máy cụ thể.

## Artifact được tạo

Pipeline fit model đúng một lần trên train rồi đánh giá độc lập cả train và test.
`classification/evaluation/outputs/` chứa 27 file:

- Root `evaluation_manifest.json` schema v2 ghi model fit split là `train`, split
  báo cáo chính là `test` và đúng 26 path trong `generated_files`.
- `train/` và `test/` mỗi thư mục có 12 artifact đánh giá cùng một child
  `evaluation_manifest.json`: 5 table CSV, 1 predictions CSV và 6 PNG.
- Predictions có 8.000 dòng cho train và 2.000 dòng cho test.

Root manifest chỉ được cập nhật sau khi cả hai split đánh giá thành công. Khi
migrate từ layout v1, pipeline dọn các bảng/hình/predictions cũ ở ngay output
root theo manifest cũ; đây là thay đổi filesystem contract có chủ đích. Pipeline
không dùng glob để xóa file ngoài danh sách do chính nó sở hữu.

## Kết quả baseline

Kết quả dùng model 100 estimator và threshold `0.5`. Test là split báo cáo chính;
train chỉ dùng để chẩn đoán overfit:

| Metric | Train — diagnostic | Test — primary |
|---|---:|---:|
| Accuracy | 0,993000 | 0,986000 |
| Precision — máy hỏng | 0,961373 | 0,900000 |
| Recall — máy hỏng | 0,826568 | 0,661765 |
| F1-score — máy hỏng | 0,888889 | 0,762712 |
| ROC-AUC | 0,996943 | 0,974211 |

Train có 8.000 mẫu (7.729 lớp 0, 271 lớp 1), confusion matrix TN=7.720,
FP=9, FN=47, TP=224. Test có 2.000 mẫu (1.932 lớp 0, 68 lớp 1), confusion
matrix TN=1.927, FP=5, FN=23, TP=45.

Chênh lệch `test - train` lần lượt là -0,007000 Accuracy, -0,061373 Precision,
-0,164803 Recall, -0,126177 F1 và -0,022732 ROC-AUC. Model nhìn chung hoạt
động tốt, nhưng gap Recall/F1 của lớp thiểu số cho thấy dấu hiệu overfit hoặc
generalization gap cần theo dõi. Không tuning threshold trên test set; test chỉ
được dùng để báo cáo khả năng tổng quát hóa.

API tích hợp hiện tại:

```python
from classification.evaluation.adapter import (
    evaluate_classification_outputs,
    evaluate_fitted_classifier,
)
from classification.evaluation.runner import run_classification_evaluation
from classification.evaluation.run_machine_failure_evaluation import (
    evaluate_machine_failure_splits,
)
```

`evaluate_classification_outputs` và `run_classification_evaluation` chỉ nhận
thư mục của một split, ví dụ `outputs/train` hoặc `outputs/test`. Không gọi hai
API generic này trực tiếp vào aggregate output root; root
`classification/evaluation/outputs/` được dành cho
`evaluate_machine_failure_splits` hoặc CLI để quản lý manifest schema v2.

Chi tiết các quyết định tích hợp nằm trong
`classification/evaluation/docs/PROJECT_INTEGRATION_NOTES.md`; lịch sử prompt và
kiểm tra AI nằm trong `classification/evaluation/docs/ai_prompting_log.md`.
