# KẾ HOẠCH THỰC HIỆN PHẦN ĐÁNH GIÁ VÀ TRỰC QUAN HÓA LIGHTGBM CLASSIFICATION

> **Cập nhật phạm vi ngày 2026-08-21:** Theo phản hồi của giảng viên được người
> dùng xác nhận, `sklearn.model_selection.train_test_split` được phép dùng riêng
> cho bước chia dữ liệu. Model, thuật toán đánh giá và metrics vẫn không được gọi
> trực tiếp từ sklearn/LightGBM. Cập nhật này thay thế các câu cấm
> `sklearn.model_selection` trong bản kế hoạch gốc; mọi lệnh cấm
> `sklearn.metrics`, metric có sẵn, `model.score()` và AUC có sẵn vẫn giữ nguyên.

> **Cập nhật layout tích hợp ngày 2026-08-21:** Phần đánh giá hiện nằm trọn trong
> `classification/evaluation/`: module ở package này, test ở
> `classification/evaluation/tests/`, artifact ở
> `classification/evaluation/outputs/` và tài liệu ở
> `classification/evaluation/docs/`. Entry point chuẩn là
> `python -m classification.evaluation.run_machine_failure_evaluation`; runner
> được import từ `classification.evaluation.runner`. `requirements.txt` và
> `.gitignore` vẫn nằm ở repository root. Các path này thay thế cấu trúc khung
> độc lập trong những phần triển khai bên dưới.

## 1. Mục tiêu

Xây dựng hoàn chỉnh phần đánh giá và trực quan hóa kết quả của bài toán phân loại sử dụng LightGBM, với các yêu cầu:

- Không sử dụng `sklearn` cho model hoặc metrics; ngoại lệ duy nhất là
  `sklearn.model_selection.train_test_split` ở bước chia dữ liệu.
- Không sử dụng các hàm metric có sẵn.
- Không sử dụng `model.score()`.
- Tự xây dựng:
  - Accuracy.
  - Precision.
  - Recall.
  - F1-score.
  - Confusion Matrix.
  - Classification Report.
  - ROC Curve.
  - ROC-AUC.
- Sử dụng:
  - `NumPy` để xử lý mảng và tính toán.
  - `Pandas` để tạo bảng và xuất CSV.
  - `Matplotlib` để trực quan hóa.
- Mỗi phần code phải có AI prompting log đi kèm.
- Prompt phải đủ chi tiết để một coding agent khác có thể tạo được chức năng tương đương.

---

## 2. Phạm vi công việc

Phần đánh giá bắt đầu sau khi thành viên xây dựng mô hình đã hoàn thành:

```python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
```

Hoặc với LightGBM native API:

```python
raw_prediction = model.predict(X_test)
```

Phần này không phụ trách:

- Làm sạch dữ liệu.
- Mã hóa dữ liệu.
- Chia Train/Test.
- Xây dựng lại thuật toán LightGBM.
- Tối ưu hyperparameter.
- Huấn luyện nhiều mô hình.

Phần này phụ trách:

1. Nhận kết quả dự đoán.
2. Kiểm tra dữ liệu đầu vào.
3. Tự xây dựng các metric.
4. Tự xây dựng Confusion Matrix.
5. Tự xây dựng ROC và ROC-AUC.
6. Tạo Classification Report.
7. Trực quan hóa kết quả.
8. Xuất toàn bộ kết quả.
9. Kiểm thử.
10. Hoàn thiện AI prompting log.

---

## 3. Quy định thư viện

### 3.1. Thư viện được phép sử dụng

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Any, Sequence
import json
import warnings
import unittest
```

### 3.2. Không được sử dụng

Không import hoặc sử dụng bất kỳ thành phần nào của `sklearn`:

```python
import sklearn
from sklearn import ...
from sklearn.metrics import ...
from sklearn.model_selection import ...
from sklearn.preprocessing import ...
```

Các hàm bị cấm gồm:

```python
accuracy_score
precision_score
recall_score
f1_score
confusion_matrix
classification_report
roc_curve
roc_auc_score
ConfusionMatrixDisplay
label_binarize
train_test_split
LabelEncoder
```

Không sử dụng:

```python
model.score(X_test, y_test)
```

Không sử dụng các hàm tính AUC có sẵn:

```python
np.trapz(...)
np.trapezoid(...)
```

AUC phải được tính thủ công bằng vòng lặp và quy tắc hình thang.

---

## 4. Lưu ý về LightGBM

`LGBMClassifier` thuộc thư viện `lightgbm`, không thuộc `sklearn`, nhưng sử dụng giao diện tương thích với scikit-learn.

Có hai trường hợp:

### Trường hợp 1: Chỉ cấm metric có sẵn của sklearn

Có thể dùng:

```python
from lightgbm import LGBMClassifier
```

Phần đánh giá chỉ nhận:

```python
y_test
y_pred
y_proba
classes
```

### Trường hợp 2: Cấm cả API kiểu sklearn

Thành viên xây dựng mô hình sử dụng:

```python
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, train_data)
```

### Phương án chốt

Module đánh giá không import `sklearn` và không import `lightgbm`.

Module chỉ nhận các mảng kết quả:

```python
y_true
y_pred
y_proba
classes
class_names
positive_label
```

Nhờ đó phần đánh giá độc lập với cách xây dựng mô hình.

---

## 5. Hợp đồng đầu vào giữa các thành viên

Thành viên xây dựng mô hình cần bàn giao:

```python
y_test
y_pred
y_proba
classes
class_names
positive_label
```

### 5.1. `y_test`

Nhãn thật của tập test:

```python
y_test.shape == (n_samples,)
```

### 5.2. `y_pred`

Nhãn dự đoán:

```python
y_pred.shape == (n_samples,)
```

### 5.3. `y_proba`

Binary classification có thể có dạng:

```python
y_proba.shape == (n_samples,)
```

hoặc:

```python
y_proba.shape == (n_samples, 2)
```

Multiclass classification:

```python
y_proba.shape == (n_samples, n_classes)
```

### 5.4. `classes`

Danh sách lớp theo đúng thứ tự cột xác suất:

```python
classes = [0, 1]
```

hoặc:

```python
classes = [0, 1, 2]
```

### 5.5. `class_names`

Tên lớp để hiển thị:

```python
class_names = ["Negative", "Positive"]
```

### 5.6. `positive_label`

Bắt buộc chỉ định đối với bài toán nhị phân:

```python
positive_label = 1
```

Không được mặc định rằng lớp dương luôn là lớp `1` nếu chưa xác minh.

---

## 6. Cấu trúc thư mục dự án

```text
machine-learning-group-6/
├── classification/
│   ├── data/raw/machine_fail.csv
│   ├── lightgbm_classification.py
│   ├── classification_metrics.py
│   ├── machine_failure_prediction.ipynb
│   └── evaluation/
│       ├── adapter.py
│       ├── runner.py
│       ├── run_machine_failure_evaluation.py
│       ├── input_validation.py
│       ├── manual_metrics.py
│       ├── manual_roc_auc.py
│       ├── reports.py
│       ├── visualizations.py
│       ├── exporters.py
│       ├── tests/
│       ├── outputs/{tables,figures,predictions}/
│       └── docs/
├── regression/
├── requirements.txt
├── .gitignore
└── README.md
```

Import tích hợp và lệnh chạy chuẩn từ repository root:

```python
from classification.evaluation.adapter import (
    evaluate_classification_outputs,
    evaluate_fitted_classifier,
)
from classification.evaluation.runner import run_classification_evaluation
```

```powershell
.\venv\Scripts\python.exe -m unittest discover -s classification/evaluation/tests -p "test_*.py" -v
.\venv\Scripts\python.exe -m classification.evaluation.run_machine_failure_evaluation
```

---

## 7. Luồng xử lý tổng thể

```text
Mô hình đã huấn luyện
        ↓
Nhận y_test, y_pred, y_proba, classes
        ↓
Kiểm tra đầu vào
        ↓
Tự xây dựng Confusion Matrix
        ↓
Tính TP, TN, FP, FN cho từng lớp
        ↓
Tính Accuracy, Precision, Recall, F1
        ↓
Tính Macro Average và Weighted Average
        ↓
Tự xây dựng ROC Curve
        ↓
Tự tính ROC-AUC
        ↓
Tạo Classification Report
        ↓
Trực quan hóa
        ↓
Lưu CSV, PNG và predictions
        ↓
Kiểm thử
        ↓
Hoàn thiện AI prompting log
```

---

# 8. Kế hoạch thực hiện chi tiết

## Bước 1. Tạo cấu hình dự án

File chính:

```text
classification/evaluation/runner.py
```

Cấu hình dự kiến:

```python
CONFIG = {
    "task_type": "binary",
    "positive_label": 1,
    "classes": [0, 1],
    "class_names": ["Negative", "Positive"],
    "output_dir": "classification/evaluation/outputs",
    "save_dpi": 300
}
```

Không hard-code các thông tin này bên trong hàm metric.

### Kết quả cần đạt

- Xác định binary hay multiclass.
- Xác định positive label.
- Xác định thứ tự lớp.
- Xác định tên lớp.
- Xác định thư mục lưu kết quả.

---

## Bước 2. Xây dựng module kiểm tra đầu vào

File:

```text
classification/evaluation/input_validation.py
```

Các hàm:

```python
validate_label_arrays(...)
validate_classes(...)
validate_class_names(...)
validate_probability_array(...)
validate_binary_configuration(...)
validate_evaluation_inputs(...)
```

### 2.1. Kiểm tra `y_true` và `y_pred`

- Là mảng một chiều.
- Có cùng số phần tử.
- Không rỗng.
- Không chứa `NaN`.
- Không chứa `Infinity`.
- Không có nhãn nằm ngoài `classes`.

### 2.2. Kiểm tra `classes`

- Không rỗng.
- Không có lớp trùng nhau.
- Chứa toàn bộ nhãn trong `y_true` và `y_pred`.
- Giữ nguyên thứ tự được cung cấp.

### 2.3. Kiểm tra `class_names`

- Số lượng tên lớp bằng số lượng lớp.
- Không có tên lớp rỗng.
- Thứ tự phải tương ứng với `classes`.

### 2.4. Kiểm tra `y_proba`

- Số dòng bằng số mẫu.
- Không có `NaN` hoặc `Infinity`.
- Xác suất nằm trong đoạn `[0, 1]`.
- Với multiclass, số cột bằng số lớp.
- Với ma trận xác suất, tổng mỗi hàng xấp xỉ bằng 1.

### 2.5. Kiểm tra binary

- Có đúng hai lớp.
- `positive_label` thuộc `classes`.
- Xác định được cột xác suất của positive class.

### Kết quả trả về

```python
{
    "n_samples": 1000,
    "n_classes": 2,
    "task_type": "binary",
    "classes": np.array([0, 1]),
    "positive_index": 1
}
```

### Điều kiện hoàn thành

- Dữ liệu sai phải raise `ValueError`.
- Thông báo lỗi phải cụ thể.
- Không tự sửa âm thầm dữ liệu sai.

---

## Bước 3. Tự xây dựng Confusion Matrix

File:

```text
classification/evaluation/manual_metrics.py
```

Hàm:

```python
build_confusion_matrix(
    y_true,
    y_pred,
    classes
)
```

Khởi tạo:

```python
matrix = np.zeros(
    (len(classes), len(classes)),
    dtype=int
)
```

Tạo ánh xạ:

```python
class_to_index = {
    label: index
    for index, label in enumerate(classes)
}
```

Duyệt từng mẫu:

```python
for true_label, predicted_label in zip(y_true, y_pred):
    true_index = class_to_index[true_label]
    predicted_index = class_to_index[predicted_label]
    matrix[true_index, predicted_index] += 1
```

Quy ước:

- Hàng: nhãn thật.
- Cột: nhãn dự đoán.

### Hàm chuẩn hóa

```python
normalize_confusion_matrix(
    confusion_matrix,
    mode="true"
)
```

Công thức:

\[
CM_{normalized}[i,j]
=
\frac{CM[i,j]}
{\sum_j CM[i,j]}
\]

Nếu tổng hàng bằng 0, toàn bộ hàng đó bằng 0.

### File đầu ra

```text
classification/evaluation/outputs/tables/confusion_matrix_counts.csv
classification/evaluation/outputs/tables/confusion_matrix_normalized.csv
```

---

## Bước 4. Tính TP, TN, FP và FN thủ công

Với mỗi lớp, xem lớp đó là Positive và các lớp còn lại là Negative.

Hàm:

```python
calculate_ovr_counts(
    confusion_matrix,
    class_index
)
```

Công thức:

\[
TP_i = CM[i,i]
\]

\[
FN_i = \sum_j CM[i,j] - TP_i
\]

\[
FP_i = \sum_j CM[j,i] - TP_i
\]

\[
TN_i = N - TP_i - FN_i - FP_i
\]

Kết quả:

```python
{
    "tp": 25,
    "tn": 60,
    "fp": 5,
    "fn": 10
}
```

Hàm phải hỗ trợ:

- Binary.
- Multiclass theo One-vs-Rest.
- Nhãn số.
- Nhãn chuỗi.

---

## Bước 5. Tự tính Accuracy

Hàm:

```python
calculate_accuracy_from_confusion_matrix(
    confusion_matrix
)
```

Công thức:

\[
Accuracy =
\frac{\sum_i CM[i,i]}
{\sum_i \sum_j CM[i,j]}
\]

Có thể thực hiện:

```python
correct = np.trace(confusion_matrix)
total = confusion_matrix.sum()
accuracy = correct / total
```

Nếu `total == 0`, raise `ValueError`.

---

## Bước 6. Tự tính Precision, Recall và F1-score

Hàm chia an toàn:

```python
safe_divide(
    numerator,
    denominator,
    undefined_value=0.0
)
```

### Precision

\[
Precision = \frac{TP}{TP + FP}
\]

### Recall

\[
Recall = \frac{TP}{TP + FN}
\]

### F1-score

\[
F1 =
2 \times
\frac{Precision \times Recall}
{Precision + Recall}
\]

Các hàm:

```python
calculate_precision(tp, fp)
calculate_recall(tp, fn)
calculate_f1_score(precision, recall)
```

Nếu mẫu số bằng 0:

- Trả về `0.0`.
- Ghi nhận trạng thái `undefined=True`.
- Có thể cảnh báo nhưng không làm chương trình dừng.

---

## Bước 7. Tính metric theo từng lớp

Hàm:

```python
calculate_per_class_metrics(
    confusion_matrix,
    classes,
    class_names=None
)
```

Kết quả cho mỗi lớp:

```python
{
    "class_label": 1,
    "class_name": "Positive",
    "tp": 25,
    "tn": 60,
    "fp": 5,
    "fn": 10,
    "precision": 0.8333,
    "recall": 0.7143,
    "f1_score": 0.7692,
    "support": 35
}
```

Trong đó:

\[
Support_i = \sum_j CM[i,j]
\]

Nên trả về:

```python
list[dict]
```

Sau đó mới chuyển sang DataFrame.

---

## Bước 8. Tính Macro Average và Weighted Average

### Macro Precision

\[
Precision_{macro}
=
\frac{1}{K}
\sum_{i=1}^{K} Precision_i
\]

### Macro Recall

\[
Recall_{macro}
=
\frac{1}{K}
\sum_{i=1}^{K} Recall_i
\]

### Macro F1

\[
F1_{macro}
=
\frac{1}{K}
\sum_{i=1}^{K} F1_i
\]

### Weighted Precision

\[
Precision_{weighted}
=
\frac{
\sum_i Support_i \times Precision_i
}{
\sum_i Support_i
}
\]

Tương tự cho Recall và F1.

Hàm:

```python
calculate_aggregate_metrics(
    per_class_metrics
)
```

Kết quả:

```python
{
    "precision_macro": ...,
    "recall_macro": ...,
    "f1_macro": ...,
    "precision_weighted": ...,
    "recall_weighted": ...,
    "f1_weighted": ...
}
```

### Metric chính để trực quan hóa

#### Binary

- Accuracy.
- Precision của positive class.
- Recall của positive class.
- F1 của positive class.
- ROC-AUC.

#### Multiclass

- Accuracy.
- Macro Precision.
- Macro Recall.
- Macro F1.
- Macro ROC-AUC OvR.

---

## Bước 9. Viết hàm đánh giá tổng hợp

Hàm:

```python
evaluate_classification(
    y_true,
    y_pred,
    classes,
    class_names=None,
    positive_label=None
)
```

Luồng xử lý:

1. Kiểm tra đầu vào.
2. Tạo Confusion Matrix.
3. Chuẩn hóa Confusion Matrix.
4. Tính Accuracy.
5. Tính metric từng lớp.
6. Tính Macro Average.
7. Tính Weighted Average.
8. Trích metric của positive class nếu là binary.
9. Trả về dictionary tổng hợp.

Kết quả:

```python
{
    "task_type": "binary",
    "accuracy": 0.91,
    "positive_label": 1,
    "precision": 0.89,
    "recall": 0.87,
    "f1_score": 0.88,
    "precision_macro": 0.90,
    "recall_macro": 0.89,
    "f1_macro": 0.895,
    "precision_weighted": 0.91,
    "recall_weighted": 0.91,
    "f1_weighted": 0.91,
    "confusion_matrix": ...,
    "normalized_confusion_matrix": ...,
    "per_class_metrics": ...
}
```

ROC-AUC được tách thành module riêng.

---

# 9. Tự xây dựng ROC Curve và ROC-AUC

File:

```text
classification/evaluation/manual_roc_auc.py
```

## 9.1. Chuyển nhãn thành binary

Hàm:

```python
convert_to_binary_targets(
    y_true,
    positive_label
)
```

Ví dụ:

```python
binary_target = np.array([
    1 if label == positive_label else 0
    for label in y_true
])
```

---

## 9.2. Trích xác suất positive class

Hàm:

```python
extract_positive_scores(
    y_proba,
    classes,
    positive_label
)
```

Nếu `y_proba` một chiều:

```python
positive_scores = y_proba
```

Nếu `y_proba` hai chiều:

```python
positive_index = classes.index(positive_label)
positive_scores = y_proba[:, positive_index]
```

---

## 9.3. Tự xây dựng ROC Curve

Hàm:

```python
calculate_binary_roc_curve(
    y_true,
    y_score,
    positive_label
)
```

Quy trình:

1. Chuyển `y_true` thành 0 và 1.
2. Sắp xếp mẫu theo `y_score` giảm dần.
3. Xác định vị trí score thay đổi.
4. Tính TP tích lũy.
5. Tính FP tích lũy.
6. Thêm điểm bắt đầu `(0, 0)`.
7. Chuyển TP thành TPR.
8. Chuyển FP thành FPR.
9. Trả về thresholds, FPR và TPR.

Công thức:

\[
TPR = \frac{TP}{P}
\]

\[
FPR = \frac{FP}{N}
\]

Trong đó:

- `P`: số mẫu positive.
- `N`: số mẫu negative.

Nếu không có positive hoặc negative sample, raise `ValueError`.

---

## 9.4. Tự tính AUC bằng quy tắc hình thang

Hàm:

```python
calculate_auc_trapezoid(
    fpr,
    tpr
)
```

Không dùng `np.trapz` hoặc `np.trapezoid`.

Cách tính:

```python
auc = 0.0

for i in range(1, len(fpr)):
    width = fpr[i] - fpr[i - 1]
    left_height = tpr[i - 1]
    right_height = tpr[i]

    area = width * (left_height + right_height) / 2
    auc += area
```

Công thức:

\[
AUC =
\sum_{i=1}^{n-1}
(FPR_{i+1}-FPR_i)
\times
\frac{TPR_{i+1}+TPR_i}{2}
\]

Kiểm tra:

```python
0.0 <= auc <= 1.0
```

---

## 9.5. ROC-AUC cho multiclass

Hàm:

```python
calculate_multiclass_roc_ovr(
    y_true,
    y_proba,
    classes,
    class_names=None
)
```

Với từng lớp:

1. Xem lớp hiện tại là Positive.
2. Các lớp còn lại là Negative.
3. Trích xác suất tương ứng.
4. Tính FPR.
5. Tính TPR.
6. Tính AUC.
7. Lưu kết quả.

Kết quả:

```python
{
    0: {
        "fpr": ...,
        "tpr": ...,
        "thresholds": ...,
        "auc": 0.91
    },
    1: {
        "fpr": ...,
        "tpr": ...,
        "thresholds": ...,
        "auc": 0.87
    }
}
```

Macro AUC:

\[
AUC_{macro}
=
\frac{1}{K}
\sum_{i=1}^{K} AUC_i
\]

Nếu một lớp không xuất hiện trong `y_test`, cần cảnh báo và đánh dấu AUC không xác định.

---

# 10. Tạo Classification Report thủ công

File:

```text
classification/evaluation/reports.py
```

Hàm:

```python
create_classification_report_dataframe(
    per_class_metrics,
    aggregate_metrics,
    accuracy
)
```

Các cột:

```text
precision
recall
f1_score
support
```

Các dòng:

```text
Class 0
Class 1
...
accuracy
macro avg
weighted avg
```

`pandas` chỉ được dùng để tổ chức dữ liệu:

```python
report_df = pd.DataFrame(rows)
```

Không sử dụng `sklearn.classification_report`.

---

# 11. Trực quan hóa kết quả

File:

```text
classification/evaluation/visualizations.py
```

Sử dụng:

```python
fig, ax = plt.subplots()
```

Không sử dụng bất kỳ công cụ trực quan hóa nào của sklearn.

---

## 11.1. Confusion Matrix số lượng

Hàm:

```python
plot_confusion_matrix(
    matrix,
    class_names,
    normalized=False,
    output_path=None
)
```

Dùng:

```python
ax.imshow(matrix)
```

Yêu cầu:

- Hiển thị ma trận.
- Đặt tick.
- Đặt tên lớp.
- Ghi số vào từng ô.
- Thêm colorbar.
- Trục X: `Predicted label`.
- Trục Y: `True label`.

File:

```text
classification/evaluation/outputs/figures/confusion_matrix_counts.png
```

---

## 11.2. Confusion Matrix chuẩn hóa

Dùng lại hàm:

```python
plot_confusion_matrix(
    normalized_matrix,
    class_names,
    normalized=True
)
```

Giá trị hiển thị:

```text
0.92
0.08
```

File:

```text
classification/evaluation/outputs/figures/confusion_matrix_normalized.png
```

---

## 11.3. Biểu đồ cột metric tổng thể

Hàm:

```python
plot_overall_metrics_bar(
    metrics,
    output_path
)
```

Binary:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
```

Multiclass:

```text
Accuracy
Macro Precision
Macro Recall
Macro F1
Macro ROC-AUC
```

Yêu cầu:

- Trục Y từ 0 đến 1.
- Hiển thị giá trị trên đầu từng cột.
- Làm tròn ba chữ số.
- Không tính lại metric trong hàm vẽ.

File:

```text
classification/evaluation/outputs/figures/overall_metrics_bar.png
```

---

## 11.4. Biểu đồ metric theo từng lớp

Hàm:

```python
plot_per_class_metrics(
    report_df,
    output_path
)
```

Mỗi lớp có ba cột:

- Precision.
- Recall.
- F1-score.

File:

```text
classification/evaluation/outputs/figures/per_class_metrics.png
```

---

## 11.5. ROC Curve

### Binary

Hàm:

```python
plot_binary_roc_curve(
    fpr,
    tpr,
    auc,
    output_path
)
```

Yêu cầu:

- Đường ROC.
- Đường tham chiếu ngẫu nhiên từ `(0,0)` đến `(1,1)`.
- AUC trong legend.
- Trục X: False Positive Rate.
- Trục Y: True Positive Rate.
- Giới hạn hai trục từ 0 đến 1.

File:

```text
classification/evaluation/outputs/figures/roc_curve.png
```

### Multiclass

Hàm:

```python
plot_multiclass_roc_curves(
    roc_results,
    class_names,
    macro_auc,
    output_path
)
```

File:

```text
classification/evaluation/outputs/figures/roc_ovr_multiclass.png
```

---

## 11.6. Pie Chart

Nội dung:

```text
Correct Predictions vs Incorrect Predictions
```

Hàm:

```python
plot_correct_incorrect_pie(
    y_true,
    y_pred,
    output_path
)
```

Tính thủ công:

```python
correct = np.sum(y_true == y_pred)
incorrect = len(y_true) - correct
```

File:

```text
classification/evaluation/outputs/figures/correct_incorrect_pie.png
```

Lưu ý:

> Pie chart chỉ minh họa tỷ lệ dự đoán đúng và sai, không thay thế Precision, Recall, F1-score hoặc ROC-AUC.

---

# 12. Xuất kết quả

File:

```text
classification/evaluation/exporters.py
```

Hàm tổng:

```python
export_evaluation_results(
    metrics,
    report_df,
    confusion_matrix,
    normalized_confusion_matrix,
    y_true,
    y_pred,
    y_proba,
    classes,
    output_dir,
    roc_results=None
)
```

## 12.1. `metrics_summary.csv`

```text
metric,value
accuracy,0.910000
precision,0.890000
recall,0.870000
f1_score,0.880000
roc_auc,0.945000
```

## 12.2. `classification_report.csv`

Bao gồm metric từng lớp, macro average và weighted average.

## 12.3. `confusion_matrix_counts.csv`

Ma trận số lượng.

## 12.4. `confusion_matrix_normalized.csv`

Ma trận chuẩn hóa.

## 12.5. `predictions.csv`

Binary:

```text
actual_label,predicted_label,positive_probability,is_correct
0,0,0.071,True
1,0,0.382,False
```

Multiclass:

```text
actual_label,predicted_label,probability_class_0,probability_class_1,probability_class_2,is_correct
```

## 12.6. `roc_points.csv`

Binary:

```text
threshold,fpr,tpr
inf,0.0,0.0
0.91,0.0,0.1
...
```

Multiclass:

```text
roc_points_class_0.csv
roc_points_class_1.csv
roc_points_class_2.csv
```

---

# 13. File tích hợp chính

File:

```text
classification/evaluation/runner.py
```

Cấu trúc:

```python
def main():
    # 1. Nhận X_test và y_test
    # 2. Nhận model đã huấn luyện
    # 3. Gọi predict
    # 4. Gọi predict_proba hoặc native predict
    # 5. Chuẩn hóa định dạng kết quả
    # 6. Validate dữ liệu
    # 7. Tính Confusion Matrix
    # 8. Tính metric thủ công
    # 9. Tính ROC-AUC thủ công
    # 10. Tạo Classification Report
    # 11. Vẽ toàn bộ biểu đồ
    # 12. Xuất toàn bộ kết quả
    # 13. In bảng tóm tắt
```

Khối chạy:

```python
if __name__ == "__main__":
    main()
```

Quy tắc:

- File tích hợp không viết lại công thức metric.
- Mọi tính toán phải gọi function từ module tương ứng.
- Không hard-code dataset, target, class hoặc output path.

---

# 14. Kế hoạch kiểm thử

Không sử dụng sklearn làm đáp án đối chiếu.

## Test 1. Binary metric cơ bản

```python
y_true = np.array([0, 0, 1, 1, 1])
y_pred = np.array([0, 1, 1, 0, 1])
```

Confusion Matrix mong đợi:

```text
[[1, 1],
 [1, 2]]
```

Với positive class là `1`:

```text
TP = 2
TN = 1
FP = 1
FN = 1
```

Kết quả mong đợi:

\[
Accuracy = 3/5 = 0.6
\]

\[
Precision = 2/3
\]

\[
Recall = 2/3
\]

\[
F1 = 2/3
\]

---

## Test 2. Dự đoán hoàn hảo

```python
y_true = np.array([0, 1, 0, 1])
y_pred = np.array([0, 1, 0, 1])
```

Kết quả:

```text
Accuracy = 1
Precision = 1
Recall = 1
F1 = 1
```

---

## Test 3. Không dự đoán được positive class

```python
y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0, 0, 0, 0])
```

Kết quả:

```text
TP = 0
FP = 0
Precision = 0
Recall = 0
F1 = 0
```

Chương trình không được lỗi chia cho 0.

---

## Test 4. Nhãn dạng chuỗi

```python
y_true = np.array(["normal", "fraud", "normal"])
y_pred = np.array(["normal", "normal", "normal"])
classes = ["normal", "fraud"]
```

Yêu cầu:

- Không phụ thuộc vào nhãn số.
- Confusion Matrix phải chính xác.

---

## Test 5. Multiclass

```python
y_true = np.array([0, 1, 2, 2])
y_pred = np.array([0, 2, 2, 1])
```

Confusion Matrix mong đợi:

```text
[[1, 0, 0],
 [0, 0, 1],
 [0, 1, 1]]
```

Kiểm tra:

- Metric từng lớp.
- Macro Average.
- Weighted Average.
- Support.

---

## Test 6. ROC-AUC với đáp án biết trước

```python
y_true = np.array([0, 0, 1, 1])
y_score = np.array([0.1, 0.4, 0.35, 0.8])
```

AUC mong đợi:

```text
0.75
```

Kiểm tra:

```python
abs(calculated_auc - 0.75) < 1e-8
```

---

## Test 7. Các score bằng nhau

```python
y_true = np.array([0, 1, 0, 1])
y_score = np.array([0.5, 0.5, 0.5, 0.5])
```

Yêu cầu:

- Xử lý đúng threshold bị trùng.
- Không phụ thuộc vào thứ tự mẫu.

---

## Test 8. Xác suất không hợp lệ

```python
y_proba = np.array([
    [1.2, -0.2],
    [0.3, 0.7]
])
```

Phải raise:

```python
ValueError
```

---

## Test 9. Sai số cột xác suất

```python
classes = [0, 1, 2]
y_proba.shape == (100, 2)
```

Phải báo lỗi số cột không khớp số lớp.

---

## Test 10. Kiểm tra file đầu ra

Sau khi chạy, trong `classification/evaluation/outputs/` phải có:

```text
metrics_summary.csv
classification_report.csv
confusion_matrix_counts.csv
confusion_matrix_normalized.csv
predictions.csv
roc_points.csv
confusion_matrix_counts.png
confusion_matrix_normalized.png
overall_metrics_bar.png
roc_curve.png
correct_incorrect_pie.png
```

---

# 15. AI Prompting Log

File:

```text
classification/evaluation/docs/ai_prompting_log.md
```

## 15.1. Quy tắc bắt buộc cho mọi prompt

Đặt đoạn sau ở đầu mỗi prompt:

```text
Yêu cầu bắt buộc:

- Không import hoặc sử dụng sklearn dưới bất kỳ hình thức nào.
- Không sử dụng sklearn.metrics, sklearn.model_selection hoặc sklearn.preprocessing.
- Không sử dụng các hàm metric có sẵn từ LightGBM.
- Không sử dụng model.score().
- Không sử dụng numpy.trapezoid hoặc numpy.trapz để tính AUC.
- Các công thức đánh giá phải được tự xây dựng bằng Python và NumPy.
- Pandas chỉ được dùng để tạo bảng và xuất file.
- Matplotlib chỉ được dùng để trực quan hóa.
- Code phải có type hint, docstring, kiểm tra lỗi và ví dụ sử dụng.
- Không hard-code số lượng lớp, tên lớp hoặc positive label.
```

---

## P01. Input validation

```text
Hãy viết file classification/evaluation/input_validation.py cho một dự án đánh giá
mô hình classification.

Cần tạo các hàm:
1. validate_label_arrays
2. validate_classes
3. validate_class_names
4. validate_probability_array
5. validate_evaluation_inputs

Yêu cầu:
- y_true và y_pred phải là mảng một chiều, cùng số mẫu và không rỗng.
- Kiểm tra NaN và infinity.
- Kiểm tra toàn bộ nhãn đều thuộc classes.
- classes không được trùng.
- Kiểm tra y_proba cho cả binary và multiclass.
- Xác suất phải nằm trong khoảng 0 đến 1.
- Với ma trận xác suất, tổng từng hàng phải xấp xỉ 1.
- Binary phải cho phép chỉ định positive_label.
- Trả về metadata gồm task_type, n_samples, n_classes và positive_index.
- Lỗi phải dùng ValueError với thông báo cụ thể.
```

---

## P02. Manual metrics

```text
Hãy viết file classification/evaluation/manual_metrics.py.

Cần tạo:
1. safe_divide
2. build_confusion_matrix
3. normalize_confusion_matrix
4. calculate_ovr_counts
5. calculate_accuracy_from_confusion_matrix
6. calculate_precision
7. calculate_recall
8. calculate_f1_score
9. calculate_per_class_metrics
10. calculate_aggregate_metrics
11. evaluate_classification

Yêu cầu:
- Confusion Matrix phải được xây dựng thủ công.
- Hàng là nhãn thật, cột là nhãn dự đoán.
- Tính TP, TN, FP và FN theo One-vs-Rest.
- Hỗ trợ binary và multiclass.
- Hỗ trợ nhãn số và nhãn chuỗi.
- Tính macro và weighted average.
- Chia cho 0 trả về 0 và ghi nhận trạng thái undefined.
- Không in trực tiếp bên trong các hàm metric.
```

---

## P03. Manual ROC-AUC

```text
Hãy viết file classification/evaluation/manual_roc_auc.py.

Cần tạo:
1. convert_to_binary_targets
2. extract_positive_scores
3. calculate_binary_roc_curve
4. calculate_auc_trapezoid
5. calculate_multiclass_roc_ovr

Yêu cầu:
- ROC phải được tự xây dựng từ threshold.
- Sắp xếp score giảm dần.
- Xử lý đúng các score bị trùng.
- Tính TP và FP tích lũy.
- Thêm điểm bắt đầu FPR=0, TPR=0.
- Tính AUC bằng vòng lặp quy tắc hình thang.
- Không dùng numpy.trapezoid hoặc numpy.trapz.
- Multiclass dùng One-vs-Rest.
- Trả về FPR, TPR, thresholds và AUC.
- Báo lỗi nếu không có positive hoặc không có negative sample.
```

---

## P04. Classification Report

```text
Hãy viết file classification/evaluation/reports.py.

Tạo hàm create_classification_report_dataframe.

Đầu vào:
- per_class_metrics
- aggregate_metrics
- accuracy

Yêu cầu:
- Tạo DataFrame gồm precision, recall, f1_score và support.
- Có một dòng cho từng lớp.
- Có macro avg và weighted avg.
- Có Accuracy.
- Không gọi sklearn.classification_report.
- Hỗ trợ tên lớp tùy chỉnh.
- Không tính lại metric trong module này.
```

---

## P05. Visualizations

```text
Hãy viết file classification/evaluation/visualizations.py.

Cần tạo:
1. plot_confusion_matrix
2. plot_overall_metrics_bar
3. plot_per_class_metrics
4. plot_binary_roc_curve
5. plot_multiclass_roc_curves
6. plot_correct_incorrect_pie

Yêu cầu:
- Dùng matplotlib theo Figure/Axes API.
- Không sử dụng ConfusionMatrixDisplay.
- Tự ghi giá trị vào từng ô Confusion Matrix.
- Có hình raw và normalized.
- Bar chart có giới hạn trục Y từ 0 đến 1.
- ROC có đường tham chiếu ngẫu nhiên.
- Pie chart thể hiện correct và incorrect predictions.
- Mỗi hàm nhận output_path.
- Lưu PNG 300 DPI bằng fig.savefig.
- Gọi plt.close(fig) sau khi lưu.
- Không tính lại các metric, trừ correct/incorrect của pie chart.
```

---

## P06. Exporter

```text
Hãy viết file classification/evaluation/exporters.py.

Tạo hàm export_evaluation_results.

Cần xuất:
1. metrics_summary.csv
2. classification_report.csv
3. confusion_matrix_counts.csv
4. confusion_matrix_normalized.csv
5. predictions.csv
6. roc_points.csv hoặc ROC riêng cho từng lớp

Yêu cầu:
- Tự tạo thư mục nếu chưa tồn tại.
- Không lưu index thừa.
- Dùng encoding UTF-8.
- Predictions phải chứa actual_label, predicted_label, xác suất và is_correct.
- Tên cột xác suất lấy từ tên lớp.
- Trả về dictionary chứa đường dẫn các file.
```

---

## P07. Integration runner

```text
Hãy viết file classification/evaluation/runner.py.

Giả định mô hình đã được thành viên khác xây dựng và huấn luyện.

Luồng:
1. Nhận model, X_test và y_test.
2. Gọi predict.
3. Nhận xác suất dự đoán.
4. Chuẩn hóa output cho binary hoặc multiclass.
5. Validate đầu vào.
6. Gọi evaluate_classification.
7. Gọi các hàm ROC-AUC.
8. Tạo classification report.
9. Vẽ toàn bộ biểu đồ.
10. Xuất toàn bộ bảng và prediction.
11. In tóm tắt cuối chương trình.

Yêu cầu:
- Không viết lại công thức metric trong file này.
- Mọi tính toán phải gọi function từ các module.
- Có hàm main.
- Có if __name__ == '__main__'.
- Không hard-code dataset, target, lớp hoặc output path.
```

---

## P08. Unit tests

```text
Hãy viết các unit test bằng thư viện unittest của Python.

Không sử dụng sklearn để đối chiếu.

Các test cần có:
1. Binary Confusion Matrix.
2. Binary Accuracy, Precision, Recall và F1 có đáp án tính tay.
3. Dự đoán hoàn hảo.
4. Không dự đoán lớp positive.
5. Nhãn dạng chuỗi.
6. Multiclass Confusion Matrix.
7. Macro và weighted average.
8. ROC-AUC có đáp án 0.75.
9. Score bị trùng.
10. Probability ngoài khoảng 0 đến 1.
11. Probability sai shape.
12. Tạo thành công các file PNG và CSV.

Sử dụng assertAlmostEqual cho metric dạng số thực.
```

---

## 15.2. Mẫu ghi một mục trong AI Prompting Log

```markdown
## P03 — Manual ROC-AUC

- Ngày thực hiện:
- Công cụ AI:
- Người thực hiện:
- File đầu ra: `classification/evaluation/manual_roc_auc.py`
- Mục tiêu: Tự xây dựng ROC Curve và ROC-AUC.

### Prompt đã sử dụng

[Dán toàn bộ prompt P03 tại đây]

### Code AI tạo

- Các hàm được tạo:
- Số dòng code:
- Thư viện được import:

### Kiểm tra điều kiện

- Không dùng sklearn: Đạt/Không đạt
- Không dùng numpy.trapezoid: Đạt/Không đạt
- Hỗ trợ binary: Đạt/Không đạt
- Hỗ trợ multiclass: Đạt/Không đạt
- Xử lý score bị trùng: Đạt/Không đạt

### Điều chỉnh thủ công

- Nội dung đã sửa:
- Lý do sửa:
- Người sửa:

### Kết quả kiểm thử

- Test AUC = 0.75: Passed/Failed
- Test score trùng: Passed/Failed
- Test thiếu positive sample: Passed/Failed

### Kết luận

- Module đã sẵn sàng tích hợp: Có/Không
```

---

# 16. Thứ tự triển khai thực tế

1. Tạo cấu trúc thư mục.
2. Hoàn thành `input_validation.py`.
3. Hoàn thành Confusion Matrix.
4. Hoàn thành TP, TN, FP, FN.
5. Hoàn thành Accuracy, Precision, Recall, F1.
6. Hoàn thành metric từng lớp.
7. Hoàn thành Macro và Weighted Average.
8. Viết test metric.
9. Hoàn thành ROC binary.
10. Hoàn thành AUC.
11. Viết test AUC bằng 0.75.
12. Mở rộng ROC multiclass.
13. Tạo Classification Report.
14. Tạo các biểu đồ.
15. Tạo exporter.
16. Tạo integration runner.
17. Chạy toàn bộ test.
18. Kiểm tra không có import sklearn.
19. Hoàn thiện AI prompting log.
20. Chạy lại toàn bộ dự án trên dữ liệu thật.

---

# 17. Kiểm tra tự động không dùng sklearn

Tìm trong toàn bộ source code:

```bash
grep -R "sklearn" .
```

Kết quả không được có import sklearn trong các file Python.

Kiểm tra các hàm có sẵn bị cấm:

```bash
grep -R "accuracy_score\|precision_score\|recall_score\|roc_auc_score\|classification_report\|ConfusionMatrixDisplay" .
```

Lưu ý:

- `f1_score` có thể là tên hàm tự xây dựng.
- Vì vậy cần kiểm tra cả dòng import, không chỉ tên hàm.

---

# 18. Sản phẩm cuối cùng cần bàn giao

## 18.1. Code

- `classification/evaluation/input_validation.py`
- `classification/evaluation/manual_metrics.py`
- `classification/evaluation/manual_roc_auc.py`
- `classification/evaluation/reports.py`
- `classification/evaluation/visualizations.py`
- `classification/evaluation/exporters.py`
- `classification/evaluation/runner.py`
- `classification/evaluation/adapter.py`
- `classification/evaluation/run_machine_failure_evaluation.py`

## 18.2. Kiểm thử

- `classification/evaluation/tests/test_manual_metrics.py`
- `classification/evaluation/tests/test_manual_roc_auc.py`
- `classification/evaluation/tests/test_integration.py`
- `classification/evaluation/tests/test_input_validation.py`
- `classification/evaluation/tests/test_classification_adapter.py`
- `classification/evaluation/tests/test_machine_failure_pipeline.py`

## 18.3. Bảng kết quả

Các bảng nằm trong `classification/evaluation/outputs/tables/`; prediction nằm
trong `classification/evaluation/outputs/predictions/`.

- `metrics_summary.csv`
- `classification_report.csv`
- `confusion_matrix_counts.csv`
- `confusion_matrix_normalized.csv`
- `predictions.csv`
- `roc_points.csv`

## 18.4. Hình trực quan hóa

Các hình nằm trong `classification/evaluation/outputs/figures/`.

- `confusion_matrix_counts.png`
- `confusion_matrix_normalized.png`
- `overall_metrics_bar.png`
- `per_class_metrics.png`
- `roc_curve.png`
- `correct_incorrect_pie.png`

## 18.5. Minh chứng AI

Tất cả tài liệu bàn giao nằm trong `classification/evaluation/docs/`.

- `classification/evaluation/docs/ai_prompting_log.md`

---

# 19. Checklist nghiệm thu cuối cùng

## Code

- [ ] Không có `import sklearn`.
- [ ] Không gọi `model.score()`.
- [ ] Không sử dụng metric có sẵn của LightGBM.
- [ ] Không dùng `numpy.trapezoid` hoặc `numpy.trapz`.
- [ ] Confusion Matrix được tự xây dựng.
- [ ] TP, TN, FP, FN được tự tính.
- [ ] Accuracy được tự tính.
- [ ] Precision được tự tính.
- [ ] Recall được tự tính.
- [ ] F1-score được tự tính.
- [ ] ROC được tự xây dựng từ score và threshold.
- [ ] AUC được tính bằng vòng lặp hình thang.
- [ ] Hỗ trợ binary.
- [ ] Hỗ trợ multiclass.
- [ ] Hỗ trợ nhãn dạng chuỗi.
- [ ] Positive label được chỉ định rõ.

## Trực quan hóa

- [ ] Confusion Matrix số lượng.
- [ ] Confusion Matrix chuẩn hóa.
- [ ] Bar chart metric tổng thể.
- [ ] Bar chart metric từng lớp.
- [ ] ROC Curve.
- [ ] Pie chart đúng/sai.
- [ ] Hình được lưu ở 300 DPI.
- [ ] Không có chữ hoặc tick bị cắt.

## Kết quả

- [ ] Có `metrics_summary.csv`.
- [ ] Có `classification_report.csv`.
- [ ] Có `confusion_matrix_counts.csv`.
- [ ] Có `confusion_matrix_normalized.csv`.
- [ ] Có `predictions.csv`.
- [ ] Có `roc_points.csv`.
- [ ] Có toàn bộ file PNG.

## Kiểm thử

- [ ] Binary metrics đúng với đáp án tính tay.
- [ ] AUC bằng 0.75 với bộ dữ liệu mẫu.
- [ ] Multiclass hoạt động.
- [ ] Không lỗi chia cho 0.
- [ ] Phát hiện xác suất sai.
- [ ] Phát hiện shape sai.
- [ ] Chạy lại toàn bộ từ đầu thành công.

## AI Prompting Log

- [ ] Mỗi module có Prompt ID.
- [ ] Lưu nguyên văn prompt.
- [ ] Ghi rõ code AI tạo.
- [ ] Ghi rõ phần con người chỉnh sửa.
- [ ] Có kết quả kiểm thử.
- [ ] Prompt đủ chi tiết để coding agent khác tạo chức năng tương đương.

---

# 20. Kết luận

Kế hoạch này bảo đảm phần đánh giá và trực quan hóa:

- Không phụ thuộc vào `sklearn`.
- Không sử dụng các hàm metric có sẵn.
- Thể hiện rõ cách xây dựng từng công thức.
- Đáp ứng yêu cầu metric dưới dạng function.
- Có đầy đủ trực quan hóa.
- Có kiểm thử.
- Có AI prompting log cho từng phần code.
- Có thể triển khai tuần tự và tích hợp với mô hình LightGBM do thành viên khác xây dựng.
