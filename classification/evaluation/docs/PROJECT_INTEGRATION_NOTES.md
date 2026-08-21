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
  `classification/evaluation/outputs/`.

## Layout tích hợp hiện tại

Toàn bộ phần đánh giá classification được gom dưới
`classification/evaluation/`:

- Các module: `adapter.py`, `runner.py`, `input_validation.py`,
  `manual_metrics.py`, `manual_roc_auc.py`, `reports.py`, `visualizations.py`,
  `exporters.py` và entry point `run_machine_failure_evaluation.py`.
- Tests: `classification/evaluation/tests/`.
- Artifacts: root manifest schema v2 và hai namespace
  `classification/evaluation/outputs/{train,test}/`.
- Tài liệu: `classification/evaluation/docs/`.

`requirements.txt` và `.gitignore` tiếp tục nằm ở repository root. Import công
khai và lệnh chạy chuẩn là:

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

```powershell
.\venv\Scripts\python.exe -m unittest discover -s classification/evaluation/tests -p "test_*.py" -v
.\venv\Scripts\python.exe -m classification.evaluation.run_machine_failure_evaluation
```

## Quyết định về sklearn

Ngày 2026-08-21, người dùng xác nhận phản hồi của giảng viên: sklearn được phép
chỉ cho bước chia dữ liệu. Vì vậy notebook và pipeline tiếp tục dùng
`sklearn.model_selection.train_test_split` với `stratify=y`.

Các giới hạn còn giữ nguyên:

- Model LightGBM-style phải là implementation của repository.
- Không dùng `sklearn.metrics`, metric của LightGBM hoặc `model.score()`.
- Confusion Matrix, Accuracy, Precision, Recall, F1, ROC và AUC được tính thủ
  công trong `classification/evaluation/`.
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
| Class counts train | `0=7.729`, `1=271` |
| Class counts test | `0=1.932`, `1=68` |
| Model fit split | `train` |
| Primary reporting split | `test` |

Dataset có 10.000 mẫu, gồm 9.661 mẫu lớp 0 và 339 mẫu lớp 1. Accuracy đơn lẻ
không đủ phản ánh chất lượng do positive rate chỉ 3,39%; cần đọc cùng Precision,
Recall, F1 và ROC-AUC của lớp dương.

## Luồng tích hợp

```text
machine_fail.csv
  -> preprocessing 6 feature không leakage
  -> sklearn train_test_split (phạm vi sklearn được cho phép)
  -> LightGBMClassification.fit trên train đúng một lần
  -> evaluate_machine_failure_splits
  -> predict/predict_proba đúng một lần cho mỗi split train/test
  -> classification.evaluation.adapter
  -> classification.evaluation.runner
  -> classification/evaluation/* (metric và ROC-AUC thủ công)
  -> classification/evaluation/outputs/{train,test}/{tables,figures,predictions}
  -> root evaluation_manifest.json schema v2
```

Adapter không tính lại metric. Nó lấy `model.classes_`, kiểm tra positive label
và ánh xạ tên lớp theo chính giá trị label trước khi gọi runner độc lập. Adapter
và generic runner chỉ được nhận output directory của một split, chẳng hạn
`outputs/train` hoặc `outputs/test`; không gọi chúng trực tiếp vào aggregate
output root vì child manifest không được phép ghi đè root manifest schema v2.

Helper public `evaluate_machine_failure_splits(model, X_train, X_test, y_train,
y_test, *, output_dir=DEFAULT_OUTPUT_DIR, save_dpi=300)` là orchestration dùng
chung cho CLI và notebook. Helper tự tạo prediction/probability cho hai split,
gọi adapter vào `output_dir/train` và `output_dir/test`, rồi mới ghi root manifest
nếu cả hai evaluation thành công. Return giữ các field top-level của test để
tương thích API cũ, đồng thời bổ sung:

- `split_results={"train": ..., "test": ...}`;
- `primary_reporting_split="test"`;
- `pipeline_manifest_path` và `manifest_path` cùng trỏ root schema-v2 manifest.

Mỗi child result vẫn có `manifest_path` riêng tới child manifest của split đó.
Train metrics chỉ là diagnostic để phát hiện overfit; test metrics là kết quả
báo cáo khả năng tổng quát hóa.

## Những file được thêm hoặc điều chỉnh

- `classification/evaluation/`: package chứa validation, manual metrics,
  manual ROC-AUC, report, hình và exporter.
- `classification/evaluation/runner.py`: điều phối đánh giá từ các mảng dự đoán.
- `classification/evaluation/adapter.py`: adapter model thật.
- `classification/evaluation/run_machine_failure_evaluation.py`: entry point
  tái lập toàn bộ run.
- `classification/classification_metrics.py`: facade tương thích, không chứa
  công thức trùng lặp.
- `classification/machine_failure_prediction.ipynb`: path portable và cell gọi
  evaluation; output chạy cũ được xóa để không mâu thuẫn với baseline có seed.
- `classification/evaluation/tests/`: generic tests, adapter contract và
  dataset/split contract.
- `classification/evaluation/outputs/`: root manifest schema v2 cùng artifact
  train/test tách biệt.
- `classification/evaluation/docs/`: kế hoạch, integration notes và prompting
  log.

## Kiểm tra bàn giao

- [x] Core generic: 83/83 test đạt trước khi nối repo.
- [x] Adapter lấy đúng thứ tự cột từ `model.classes_`.
- [x] Xác suất có shape `(n, 2)`, tổng từng hàng bằng 1 và prediction khớp
  threshold.
- [x] Dataset và stratified split được kiểm tra bằng dữ liệu thật.
- [x] Chạy toàn bộ test suite sau chỉnh sửa cuối: 97/97 test đạt.
- [x] Gom module, tests, outputs và docs vào `classification/evaluation/`; cập
  nhật import, notebook, CLI và lệnh test theo namespace mới.
- [x] Fit pipeline 100 estimator một lần rồi đánh giá train 8.000 và test 2.000
  mẫu; test là primary reporting split.
- [x] Kiểm tra trực quan 12 PNG ở 300 DPI và schema/số dòng CSV của hai split.
- [x] Root manifest schema v2 liệt kê đúng 26 generated files; toàn cây output
  có 27 file tính cả root manifest.
- [x] Đối chiếu test artifacts với baseline v1: byte-identical; train artifacts
  và root schema v2 được xác minh theo contract mới.
- [x] Snapshot P10-P11 trước mở rộng train/test đã được commit và push.
- [x] P12 đã được commit, push nhánh tích hợp và fast-forward vào `LightGBM`
  sau khi toàn bộ QA đạt.

Baseline train: Accuracy 0,993000; Precision 0,961373; Recall 0,826568;
F1-score 0,888889; ROC-AUC 0,996943. Confusion matrix có TN=7.720, FP=9,
FN=47, TP=224; `train/predictions/predictions.csv` có 8.000 dòng với class
counts `0=7.729`, `1=271`.

Baseline test — kết quả báo cáo chính: Accuracy 0,986000; Precision 0,900000;
Recall 0,661765; F1-score 0,762712; ROC-AUC 0,974211. Confusion matrix có
TN=1.927, FP=5, FN=23, TP=45; `test/predictions/predictions.csv` có 2.000 dòng
với class counts `0=1.932`, `1=68`.

Gap `test - train`: Accuracy -0,007000; Precision -0,061373; Recall -0,164803;
F1 -0,126177; ROC-AUC -0,022732. Model hoạt động tốt về tổng thể, nhưng Recall
và F1 của lớp thiểu số giảm đáng kể trên test, cho thấy dấu hiệu overfit hoặc
generalization gap. Không tuning threshold trên test set.

Output hiện có 27 file: root manifest cộng với hai child tree, mỗi tree gồm 12
artifact và một child manifest. Root `generated_files` có đúng 26 path.

Môi trường sinh artifact: Python 3.13.5, NumPy 2.5.2, pandas 3.0.5,
Matplotlib 3.11.1 và scikit-learn 1.9.0. `requirements.txt` dùng lower bounds để
không khóa người dùng vào đúng môi trường này; test artifacts byte-identical với
baseline trước khi đổi layout đã được xác minh trong môi trường nêu trên.

## Lịch sử GitHub

- Nhánh tích hợp đã push: `classification-evaluation-integration`.
- Commit triển khai chính: `cd3913a` —
  `feat(classification): add manual evaluation pipeline`.
- Commit tái cấu trúc: `f32c3d2` —
  `refactor(classification): isolate evaluation package`.
- Theo xác nhận của người dùng, nhánh `LightGBM` đã được fast-forward từ
  `32397d0` qua `f32c3d2`; không force-push và không tạo pull request trung gian.
- Commit tài liệu trạng thái bàn giao này được đẩy tiếp lên cả `LightGBM` và
  `classification-evaluation-integration` để hai nhánh cùng trỏ tới snapshot cuối.
- Nhánh tích hợp train/test: `classification-train-test-evaluation`.
- Commit mở rộng train/test: `2d40939` —
  `feat(classification): evaluate train and test splits`.
- Sau QA, `LightGBM` đã được fast-forward từ `a5050fa` qua `2d40939`; không
  force-push và không tạo merge commit trung gian.
- Commit tài liệu trạng thái này được đẩy tiếp lên cả nhánh train/test và
  `LightGBM` để hai nhánh cùng trỏ tới snapshot bàn giao cuối.

## Giới hạn chủ ý

- Không tuning threshold trên test set; baseline giữ `0.5` như model.
- Test là primary reporting split; train chỉ dùng diagnostic overfit và không
  thay thế đánh giá khả năng tổng quát hóa.
- Không mở rộng thay đổi sang regression vì sklearn được phép cho data split.
- Model hiện chỉ hỗ trợ đúng hai lớp.
- Root manifest schema v2 thay layout root-level `tables/`, `figures/`,
  `predictions/` cũ bằng namespace `train/` và `test/`. Đây là breaking
  filesystem contract có chủ đích; migration chỉ xóa artifact v1 đã được
  manifest cũ ghi nhận, không dùng glob để xóa file ngoài quyền sở hữu.
