# Ghi chú tích hợp đánh giá LightGBM Classification

Tệp này ghi lại các quyết định nối bộ đánh giá thủ công với repository chính.
Nó thay cho các giả định tạm thời trước khi model và dataset được cung cấp.

## Phạm vi repository

- Repository: `thanhdoan252024-hash/machine-learning-group-6`.
- Nhánh nền: `LightGBM`, tại commit `32397d0` khi bắt đầu tích hợp.
- Nhánh triển khai: `classification-evaluation-integration`.
- Model thật: `classification/lightgbm_classification.py`.
- Notebook thật: `classification/machine_failure_prediction.ipynb`.
- Dataset thật: `classification/data/raw/machine_fail.csv`.
- Target: `Machine failure`.
- Output được phép commit: toàn bộ CSV, PNG và manifest trong
  `classification/outputs/`.

## Quyết định về sklearn

Ngày 2026-08-21, người dùng xác nhận phản hồi của giảng viên: sklearn được phép
chỉ cho bước chia dữ liệu. Vì vậy notebook và pipeline tiếp tục dùng
`sklearn.model_selection.train_test_split` với `stratify=y`.

Các giới hạn còn giữ nguyên:

- Model LightGBM-style phải là implementation của repository.
- Không dùng `sklearn.metrics`, metric của LightGBM hoặc `model.score()`.
- Confusion Matrix, Accuracy, Precision, Recall, F1, ROC và AUC được tính thủ
  công trong `evaluation/`.
- Không dùng `numpy.trapz` hoặc `numpy.trapezoid` để tính AUC.

## Hợp đồng model và dataset đã xác minh

| Thuộc tính | Giá trị thật |
|---|---|
| Loại bài toán | Binary classification |
| Ordered classes | Lấy từ `model.classes_`, dữ liệu hiện tại cho `[0, 1]` |
| Positive label | `1` — máy hỏng |
| Class name của `0` | `Không hỏng máy` |
| Class name của `1` | `Hỏng máy` |
| `predict()` | Nhãn một chiều, dùng `model.threshold` |
| `predict_proba()` | Shape `(n_samples, 2)`, cột theo `model.classes_` |
| Threshold baseline | `0.5` |
| Train/test split | 80/20, stratified, `random_state=42` |
| Số mẫu train/test | 8.000 / 2.000 |

Dataset có 10.000 mẫu, gồm 9.661 mẫu lớp 0 và 339 mẫu lớp 1. Accuracy đơn lẻ
không đủ phản ánh chất lượng do positive rate chỉ 3,39%; cần đọc cùng Precision,
Recall, F1 và ROC-AUC của lớp dương.

## Luồng tích hợp

```text
machine_fail.csv
  -> preprocessing 6 feature không leakage
  -> sklearn train_test_split (phạm vi sklearn được cho phép)
  -> LightGBMClassification.fit/predict/predict_proba
  -> classification_evaluation_adapter
  -> experiments.run_classification_evaluation
  -> evaluation/* (metric và ROC-AUC thủ công)
  -> classification/outputs/{tables,figures,predictions}
```

Adapter không tính lại metric. Nó lấy `model.classes_`, kiểm tra positive label
và ánh xạ tên lớp theo chính giá trị label trước khi gọi runner độc lập.

## Những file được thêm hoặc điều chỉnh

- `evaluation/`: validation, manual metrics, manual ROC-AUC, report, hình và
  exporter.
- `experiments/run_classification_evaluation.py`: điều phối đánh giá từ các mảng
  dự đoán.
- `classification/classification_evaluation_adapter.py`: adapter model thật.
- `classification/machine_failure_pipeline.py`: entry point tái lập toàn bộ run.
- `classification/classification_metrics.py`: facade tương thích, không chứa
  công thức trùng lặp.
- `classification/machine_failure_prediction.ipynb`: path portable và cell gọi
  evaluation; output chạy cũ được xóa để không mâu thuẫn với baseline có seed.
- `tests/`: generic tests, adapter contract và dataset/split contract.
- `classification/outputs/`: artifact chạy thật được commit theo yêu cầu.

## Kiểm tra bàn giao

- [x] Core generic: 83/83 test đạt trước khi nối repo.
- [x] Adapter lấy đúng thứ tự cột từ `model.classes_`.
- [x] Xác suất có shape `(n, 2)`, tổng từng hàng bằng 1 và prediction khớp
  threshold.
- [x] Dataset và stratified split được kiểm tra bằng dữ liệu thật.
- [x] Chạy toàn bộ test suite sau chỉnh sửa cuối: 90/90 test đạt.
- [x] Chạy pipeline 100 estimator trên đủ test split 2.000 mẫu.
- [x] Kiểm tra trực quan sáu PNG ở 300 DPI và schema/số dòng CSV.
- [x] Chạy lại pipeline và đối chiếu SHA-256: toàn bộ artifact tái lập byte-for-byte.
- [ ] Ghi kết quả cuối, commit và push nhánh tích hợp.

Kết quả baseline: Accuracy 0,986000; Precision 0,900000; Recall 0,661765;
F1-score 0,762712; ROC-AUC 0,974211. Confusion matrix có TN=1.927, FP=5,
FN=23 và TP=45. `predictions.csv` có đúng 2.000 dòng dữ liệu và bốn cột.
Xác suất được lưu với 12 chữ số có nghĩa; tính lại ROC-AUC trực tiếp từ
`predictions.csv` cho 0,974211423700, khớp kết quả trước khi làm tròn bảng.

Môi trường sinh artifact: Python 3.13.5, NumPy 2.5.2, pandas 3.0.5,
Matplotlib 3.11.1 và scikit-learn 1.9.0. `requirements.txt` dùng lower bounds để
không khóa người dùng vào đúng môi trường này; khả năng tái lập byte-for-byte đã
được xác minh trong môi trường nêu trên.

## Giới hạn chủ ý

- Không tuning threshold trên test set; baseline giữ `0.5` như model.
- Không mở rộng thay đổi sang regression vì sklearn được phép cho data split.
- Model hiện chỉ hỗ trợ đúng hai lớp.
- Manifest chỉ xóa artifact cũ mà lần chạy trước của pipeline đã ghi nhận; không
  dùng glob để xóa file ngoài quyền sở hữu.
