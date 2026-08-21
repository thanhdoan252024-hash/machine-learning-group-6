# AI Prompting Log — Đánh giá LightGBM Classification thủ công

## Thông tin chung

- Ngày lập log: 2026-08-20; cập nhật kết nối repo ngày 2026-08-21
- Công cụ AI: OpenAI Codex
- Người thực hiện code và ghi log: Codex theo yêu cầu của người dùng
- Tài liệu đặc tả:
  `classification/evaluation/docs/Ke_hoach_danh_gia_truc_quan_LightGBM_Classification_khong_sklearn.md`
- Phạm vi hiện tại: đã kết nối lõi đánh giá với model, notebook và dataset thật
  của repo `machine-learning-group-6`, đồng thời sinh artifact từ test split thật.
- Ngoại lệ đã được giảng viên xác nhận qua người dùng: sklearn được phép dùng
  riêng cho chia dữ liệu; model và toàn bộ metrics không gọi trực tiếp thư viện.
- Chi tiết hợp đồng và kết quả tích hợp nằm trong
  `classification/evaluation/docs/PROJECT_INTEGRATION_NOTES.md`.

Các path trong P01-P08 phản ánh layout độc lập tại thời điểm tạo core. Layout
hiện tại, các mục P09-P11 và mọi lệnh bàn giao dùng package
`classification/evaluation/`.

Các prompt dưới đây được lưu đầy đủ theo từng Prompt ID. Chúng kết hợp yêu cầu
bắt buộc của mục 15 trong kế hoạch với hợp đồng đầu ra thực tế của code hiện có,
để một coding agent khác có thể tái tạo chức năng tương đương. Phần “Điều chỉnh
thủ công” ghi rõ các quyết định bổ sung hoặc khác biệt so với prompt khung trong
kế hoạch; không gán cho con người những thay đổi không có lịch sử xác nhận.

## Trạng thái kiểm thử được ghi nhận tại thời điểm lập log

| Nhóm kiểm thử | Tệp | Trạng thái hiện biết |
|---|---|---:|
| Input validation | `classification/evaluation/tests/test_input_validation.py` | 42/42 passed |
| Manual metrics | `classification/evaluation/tests/test_manual_metrics.py` | 19/19 passed |
| Manual ROC-AUC | `classification/evaluation/tests/test_manual_roc_auc.py` | 15/15 passed |
| Report, visualizations, exporter và runner | `classification/evaluation/tests/test_integration.py` | 7/7 passed |
| Adapter model thật | `classification/evaluation/tests/test_classification_adapter.py` | 5/5 passed |
| Dataset và stratified split | `classification/evaluation/tests/test_machine_failure_pipeline.py` | 2/2 passed |

Tổng trạng thái lõi sau QA ban đầu là 83/83 test passed. Sau khi kết nối repo,
suite có thêm 5 adapter tests và 2 dataset/split tests, đạt 90/90. Pipeline thật
đã đánh giá 2.000 mẫu và tạo đủ CSV/PNG/manifest trong
`classification/evaluation/outputs/`.

---

## P01 — Input validation

- Ngày thực hiện: 2026-08-20
- Công cụ AI: OpenAI Codex
- Người thực hiện: Codex
- File đầu ra: `evaluation/input_validation.py`
- Mục tiêu: Kiểm tra toàn bộ hợp đồng đầu vào trước khi tính metric, ROC, report hoặc output.

### Prompt đã sử dụng

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

Hãy viết file evaluation/input_validation.py cho phần đánh giá classification
độc lập với model. Chỉ dùng Python standard library và NumPy trong module này.

Tạo các hàm public:
1. validate_label_arrays(y_true, y_pred)
2. validate_classes(classes, y_true=None, y_pred=None)
3. validate_class_names(class_names, classes)
4. validate_probability_array(y_proba, n_samples, n_classes, task_type)
5. validate_binary_configuration(classes, positive_label)
6. validate_evaluation_inputs(y_true, y_pred, y_proba, classes,
   class_names=None, positive_label=None, task_type=None)

Chi tiết bắt buộc:
- y_true và y_pred phải chuyển được thành mảng một chiều, không rỗng, cùng số
  mẫu. Giữ nguyên kiểu scalar của nhãn bằng object array để nhãn số hoặc chuỗi
  không bị ép kiểu ngoài ý muốn.
- Chỉ chấp nhận nhãn scalar là chuỗi, integer Python/NumPy hoặc float hữu hạn.
  Phải hỗ trợ integer Python có độ lớn tùy ý, không ép qua float khiến overflow.
  Từ chối NaN, infinity, cấu trúc lồng và kiểu không được hỗ trợ bằng ValueError
  có thông báo nêu tên biến/vị trí lỗi.
- classes phải một chiều, không rỗng, duy nhất và giữ đúng thứ tự người gọi
  cung cấp. Mọi nhãn quan sát trong y_true/y_pred phải thuộc classes.
- class_names=None thì suy ra bằng str(label). Nếu được truyền vào, số tên phải
  bằng số lớp; mọi tên phải là chuỗi không rỗng sau khi strip.
- Binary có đúng hai lớp và bắt buộc positive_label thuộc classes. Hàm
  validate_binary_configuration trả về index của positive_label.
- task_type chỉ nhận None, binary hoặc multiclass. Nếu None, suy ra binary khi
  có hai lớp và multiclass khi có từ ba lớp; kiểm tra tính nhất quán giữa loại
  bài toán và số lớp. Nếu task là multiclass mà positive_label khác None thì
  phải báo ValueError vì positive_label chỉ có ý nghĩa với binary.
- Binary y_proba chấp nhận shape (n_samples,) — hiểu là xác suất của
  positive_label — hoặc (n_samples, 2) theo đúng thứ tự classes.
- Multiclass y_proba bắt buộc shape (n_samples, n_classes).
- Xác suất phải là số thực hữu hạn trong [0, 1]. Với y_proba hai chiều, tổng
  mỗi hàng phải xấp xỉ 1 theo np.isclose/allclose. Không clip, reshape,
  renormalize hoặc âm thầm sửa dữ liệu không hợp lệ.
- validate_evaluation_inputs trả dictionary gồm n_samples, n_classes,
  task_type, classes, class_names, positive_label, positive_index, y_true,
  y_pred và y_proba đã được kiểm tra.
- Tất cả lỗi hợp đồng dùng ValueError với thông báo cụ thể.
- Thêm type hint, docstring cho public function, __all__ và ví dụ gọi ngắn trong
  tài liệu sử dụng nếu không đặt ví dụ trực tiếp trong module.
```

### Code AI tạo

- Các hàm public: `validate_label_arrays`, `validate_classes`,
  `validate_class_names`, `validate_probability_array`,
  `validate_binary_configuration`, `validate_evaluation_inputs`.
- Helper nội bộ: chuyển object array một chiều, kiểm tra scalar label, so sánh
  label an toàn, kiểm tra membership và phân giải `task_type`.
- Số dòng sau QA cuối: 490.
- Thư viện import: `typing`, `numpy`; không dùng `math` hoặc `numbers`.
- Đầu ra tổng hợp bao gồm cả metadata cấu hình và các array đã kiểm tra; dữ liệu
  xác suất sai không bị sửa âm thầm.

### Kiểm tra điều kiện

- Không dùng sklearn hoặc metric LightGBM: Đạt.
- Không hard-code class/positive label: Đạt.
- Hỗ trợ binary vector và binary matrix: Đạt.
- Hỗ trợ multiclass matrix: Đạt.
- Giữ thứ tự classes và kiểm tra toàn bộ nhãn: Đạt.
- Kiểm tra NaN, infinity, range và row sum: Đạt.
- Hỗ trợ arbitrary-size Python integer label mà không overflow: Đạt.
- Từ chối `positive_label` trong cấu hình multiclass: Đạt.
- Type hint, docstring và thông báo ValueError cụ thể: Đạt.
- Ví dụ gọi trực tiếp trong module: Chưa có; hợp đồng và lệnh chạy được mô tả trong `README.md`.

### Điều chỉnh thủ công/điều chỉnh so với prompt khung

- Bổ sung `validate_binary_configuration`, dù danh sách hàm P01 rút gọn trong
  mục 15 không liệt kê hàm này; đây là yêu cầu rõ ràng ở bước 2.5 của kế hoạch.
- Trả thêm các array đã chuẩn hóa về representation trong metadata để runner
  không phải chuyển đổi lại.
- Dùng phép so sánh scalar an toàn thay vì phụ thuộc hoàn toàn vào equality của
  NumPy, nhằm tránh kết quả boolean array mơ hồ.
- QA bỏ việc phân loại số qua `numbers.Real`/`math` và kiểm tra riêng integer,
  float để nhãn integer cực lớn không bị ép sang floating point.
- QA bổ sung lỗi rõ ràng khi `positive_label` được truyền cho multiclass.
- Không ghi nhận một đợt chỉnh tay độc lập ngoài việc đối chiếu code với kế hoạch.

### Kết quả kiểm thử

- File kiểm thử: `tests/test_input_validation.py`.
- Trạng thái cuối: 42/42 test passed.
- Phạm vi gồm shape/length, nhãn số và chuỗi, class order/uniqueness, class
  names, probability binary/multiclass, range, NaN/infinity, row sum,
  positive label, task inference và metadata tổng hợp.
- Edge case QA mới: nhãn integer Python có độ lớn tùy ý được chấp nhận; truyền
  `positive_label` cho multiclass bị từ chối.

### Kết luận

- Module lõi sẵn sàng để các phần metric/runner sử dụng: Có.
- Cần sửa khi nối repo chính: Chỉ khi output model thực tế không đáp ứng hợp đồng đã ghi.

---

## P02 — Manual metrics

- Ngày thực hiện: 2026-08-20
- Công cụ AI: OpenAI Codex
- Người thực hiện: Codex
- File đầu ra: `evaluation/manual_metrics.py`
- Mục tiêu: Tự xây dựng Confusion Matrix và toàn bộ metric dựa trên nhãn, không dùng metric có sẵn.

### Prompt đã sử dụng

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

Hãy viết evaluation/manual_metrics.py và tái sử dụng validation từ
evaluation/input_validation.py. Không in ra màn hình trong bất kỳ hàm metric nào.

Tạo các hàm public:
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

Yêu cầu triển khai:
- Confusion Matrix được dựng thủ công bằng cách ánh xạ classes sang index rồi
  duyệt từng cặp (true, predicted). Hàng là nhãn thật, cột là nhãn dự đoán và
  thứ tự trục đúng thứ tự classes. Kết quả là ma trận integer n_classes x n_classes.
- normalize_confusion_matrix chỉ hỗ trợ mode='true': chia từng ô cho tổng hàng;
  hàng có support bằng 0 giữ toàn số 0.
- calculate_ovr_counts cho một class_index phải tính thủ công TP, TN, FP, FN từ
  Confusion Matrix theo One-vs-Rest.
- Accuracy = tổng đường chéo / tổng tất cả mẫu. Từ chối ma trận tổng bằng 0.
- Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R).
- safe_divide mặc định trả 0.0 khi denominator bằng 0. Cho phép
  return_status=True để trả tuple (value, undefined), nhờ đó metric từng lớp
  vừa có giá trị số ổn định vừa lưu cờ undefined.
- calculate_per_class_metrics trả list theo đúng thứ tự lớp. Mỗi item gồm
  class_label, class_name, tp, tn, fp, fn, precision, recall, f1_score,
  support, precision_undefined, recall_undefined và f1_undefined.
- calculate_aggregate_metrics tự tính arithmetic macro average và support-
  weighted average cho precision, recall, f1_score. Không dùng np.average như
  một metric thay thế; công thức phải rõ ràng và kiểm tra tổng support > 0.
- evaluate_classification điều phối validation, Confusion Matrix, normalized
  matrix, Accuracy, per-class metrics và aggregate metrics. Với binary, bắt
  buộc positive_label và expose precision/recall/f1 của đúng positive class.
  Với multiclass, không nhận positive_label.
- Hỗ trợ nhãn số và chuỗi, không giả định hai lớp là 0/1.
- Kiểm tra ma trận vuông, hữu hạn, không âm, chứa integer counts; kiểm tra scalar
  và rate hợp lệ. Lỗi dùng ValueError cụ thể.
- Thêm type hint, docstring, __all__ và ví dụ có đáp án tính tay trong tài liệu/test.
```

### Code AI tạo

- Các hàm public: đủ 11 hàm trong prompt.
- Số dòng sau QA cuối: 474.
- Thư viện import: `collections.abc`, `typing`, `numpy` và các hàm validation nội bộ.
- `safe_divide` có overload cho dạng trả scalar và tuple trạng thái.
- `evaluate_classification` không tính ROC; ROC được tách riêng để giữ đúng trách nhiệm module.

### Kiểm tra điều kiện

- Confusion Matrix tự xây dựng, đúng quy ước hàng thật/cột dự đoán: Đạt.
- TP/TN/FP/FN One-vs-Rest tự tính: Đạt.
- Accuracy, Precision, Recall, F1 tự tính: Đạt.
- Macro và weighted average tự tính: Đạt.
- Zero denominator trả 0.0 và có cờ undefined: Đạt.
- Nhãn số, nhãn chuỗi, binary và multiclass: Đạt.
- Không dùng sklearn/LightGBM metric/model.score: Đạt.
- Không in trong hàm metric: Đạt.
- Type hint, docstring và kiểm tra lỗi: Đạt.

### Điều chỉnh thủ công/điều chỉnh so với prompt khung

- Thiết kế `safe_divide(..., return_status=True)` để tương thích cả yêu cầu API
  trả số đơn giản và yêu cầu lưu trạng thái undefined.
- `normalize_confusion_matrix` cố ý chỉ nhận `mode='true'` để tránh mơ hồ về quy
  ước chuẩn hóa.
- Kết quả binary expose metric của lớp được chỉ định bởi `positive_label`, không
  mặc định cột cuối hoặc label `1`.
- Không ghi nhận một đợt chỉnh tay độc lập ngoài việc đối chiếu code với kế hoạch.

### Kết quả kiểm thử

- File kiểm thử: `tests/test_manual_metrics.py`.
- Trạng thái cuối: 19/19 test passed.
- Đáp án tính tay chính: ma trận `[[1, 1], [1, 2]]`, Accuracy `3/5`, và
  Precision/Recall/F1 của positive class đều bằng `2/3`.
- Phạm vi khác: perfect prediction, không dự đoán positive, string labels,
  multiclass, macro/weighted average, row rỗng và các input lỗi.

### Kết luận

- Module metric thủ công sẵn sàng tích hợp: Có.

---

## P03 — Manual ROC-AUC

- Ngày thực hiện: 2026-08-20
- Công cụ AI: OpenAI Codex
- Người thực hiện: Codex
- File đầu ra: `evaluation/manual_roc_auc.py`
- Mục tiêu: Tự xây dựng ROC Curve và ROC-AUC cho binary và multiclass One-vs-Rest.

### Prompt đã sử dụng

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

Hãy viết evaluation/manual_roc_auc.py độc lập với model.

Tạo các hàm public:
1. convert_to_binary_targets
2. extract_positive_scores
3. calculate_binary_roc_curve
4. calculate_auc_trapezoid
5. calculate_multiclass_roc_ovr

Chi tiết thuật toán:
- convert_to_binary_targets nhận y_true và positive_label, trả int array có 1
  khi label bằng positive_label và 0 cho các label còn lại. Hỗ trợ nhãn chuỗi.
- extract_positive_scores: nếu y_proba một chiều, coi trực tiếp là xác suất của
  positive_label; nếu hai chiều, tìm positive_label trong classes và lấy đúng
  cột theo thứ tự classes. Kiểm tra shape, numeric, finite và [0, 1].
- Binary ROC phải được xây dựng thủ công: kiểm tra y_true/y_score một chiều cùng
  chiều dài, đếm positive/negative, báo ValueError nếu thiếu một nhóm, stable
  sort score giảm dần, gom toàn bộ sample có score trùng thành một threshold,
  cập nhật TP/FP tích lũy theo nhóm rồi tính TPR/FPR.
- Chèn điểm bắt đầu FPR=0, TPR=0 với threshold=+infinity. Điểm cuối phải đạt
  (1,1) khi cả hai nhóm tồn tại.
- calculate_auc_trapezoid nhận FPR/TPR cùng độ dài, hữu hạn, trong [0,1], FPR
  không giảm và có ít nhất hai điểm. Tính tổng diện tích bằng vòng lặp:
  width * (left_height + right_height) / 2. Không gọi np.trapz,
  np.trapezoid hoặc hàm AUC bên ngoài.
- calculate_binary_roc_curve trả dictionary có fpr, tpr, thresholds và auc.
- Multiclass dùng One-vs-Rest cho từng class theo đúng cột y_proba. Trả per_class,
  macro_auc và undefined_classes. Mỗi class result chứa label/name, defined,
  reason, fpr, tpr, thresholds và auc.
- Nếu một lớp multiclass không có positive hoặc negative sample, phát
  RuntimeWarning, đánh dấu defined=False, auc=None và array điểm rỗng; loại lớp
  đó khỏi macro average thay vì làm hỏng các lớp hợp lệ. Nếu không có lớp nào
  xác định được, macro_auc=None.
- Kiểm tra classes duy nhất, label hợp lệ, y_proba đúng shape, hữu hạn, trong
  [0,1], và tổng hàng xấp xỉ 1.
- Thêm type hint, docstring, __all__ và ví dụ/test AUC có đáp án biết trước 0.75.
```

### Code AI tạo

- Các hàm public: đủ 5 hàm trong prompt.
- Helper nội bộ: chuẩn hóa vector/score, so sánh label an toàn, kiểm tra classes và tìm index.
- Số dòng sau QA cuối: 396.
- Thư viện import: `typing`, `warnings`, `numpy`.

### Kiểm tra điều kiện

- Không dùng sklearn: Đạt.
- Không dùng `numpy.trapezoid` hoặc `numpy.trapz`: Đạt.
- Binary ROC được sweep theo threshold tự viết: Đạt.
- Score trùng được gom thành một nhóm: Đạt.
- AUC dùng vòng lặp quy tắc hình thang: Đạt.
- Hỗ trợ multiclass One-vs-Rest: Đạt.
- Thiếu positive/negative ở binary báo ValueError: Đạt.
- Lớp multiclass không xác định được cảnh báo và loại khỏi macro: Đạt.

### Điều chỉnh thủ công/điều chỉnh so với prompt khung

- Làm rõ hai chính sách thiếu nhóm: binary trực tiếp báo lỗi; multiclass giữ kết
  quả các lớp còn lại và ghi `defined=False` cho lớp bị thiếu.
- Dùng stable sort và xử lý score bằng nhóm để AUC không phụ thuộc thứ tự ban đầu
  của các sample đồng điểm.
- `thresholds[0]` là positive infinity để biểu diễn trạng thái chưa dự đoán mẫu
  nào là positive.
- Không ghi nhận một đợt chỉnh tay độc lập ngoài việc đối chiếu code với kế hoạch.

### Kết quả kiểm thử

- File kiểm thử: `tests/test_manual_roc_auc.py`.
- Trạng thái cuối: 15/15 test passed.
- AUC biết trước: `y_true=[0,0,1,1]`, `score=[0.1,0.4,0.35,0.8]` cho AUC `0.75`.
- Score đồng hạng hoàn toàn cho AUC `0.5`; perfect ranking cho AUC `1.0`.
- Phạm vi multiclass gồm perfect OVR, lớp vắng mặt có warning, macro loại lớp
  undefined, và probability contract không hợp lệ.

### Kết luận

- Module ROC-AUC thủ công sẵn sàng tích hợp: Có.

---

## P04 — Classification Report

- Ngày thực hiện: 2026-08-20
- Công cụ AI: OpenAI Codex
- Người thực hiện: Codex
- File đầu ra: `evaluation/reports.py`
- Mục tiêu: Chỉ tổ chức các metric đã tính thành DataFrame, không tính metric lại.

### Prompt đã sử dụng

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

Hãy viết evaluation/reports.py với hàm public:
create_classification_report_dataframe(per_class_metrics, aggregate_metrics, accuracy).

Yêu cầu:
- Chỉ tổ chức metric đã có; tuyệt đối không tính lại Precision, Recall hoặc F1
  và không gọi sklearn.classification_report.
- DataFrame có schema cố định theo thứ tự cột:
  label, row_type, precision, recall, f1_score, support.
- Thêm một dòng cho mỗi item trong per_class_metrics, giữ đúng thứ tự đầu vào.
  Tên label ưu tiên class_name, fallback class_label rồi Class {index}; đặt
  row_type='class' cho mọi dòng lớp.
- Thêm các dòng theo đúng thứ tự: accuracy, macro avg, weighted avg.
- Các dòng tổng hợp đặt row_type='summary'. Không nhận diện dòng tổng hợp chỉ
  bằng text của label, vì tên một lớp hợp lệ có thể là accuracy, macro avg hoặc
  weighted avg.
- Dòng accuracy đặt giá trị Accuracy vào f1_score, precision/recall là NaN và
  support là tổng support của các lớp.
- Dòng macro avg lấy precision_macro, recall_macro, f1_macro; weighted avg lấy
  precision_weighted, recall_weighted, f1_weighted. Hai dòng dùng tổng support.
- Kiểm tra per_class_metrics không rỗng, đủ field; metric hữu hạn trong [0,1];
  support là integer không âm; aggregate đủ key; tên hiển thị không rỗng.
- Dùng Pandas chỉ để tạo bảng và NumPy để biểu diễn/kiểm tra NaN, finite.
- Thêm type hint, docstring, thông báo ValueError rõ ràng và ví dụ/test kiểm tra
  chính xác thứ tự các dòng.
```

### Code AI tạo

- Hàm public: `create_classification_report_dataframe`.
- Helper nội bộ: `_validate_unit_interval`, `_validate_support`.
- Hằng số schema: `REPORT_COLUMNS`.
- Số dòng sau QA cuối: 165.
- Thư viện import: `typing`, `numpy`, `pandas`.

### Kiểm tra điều kiện

- Không gọi classification report của sklearn: Đạt.
- Không tính lại Precision/Recall/F1: Đạt.
- Có row theo lớp, Accuracy, macro avg, weighted avg: Đạt.
- Hỗ trợ tên lớp tùy chỉnh từ dữ liệu per-class: Đạt.
- Phân biệt class/summary bằng `row_type`, kể cả khi label trùng text reserved: Đạt.
- Schema và validation ổn định: Đạt.

### Điều chỉnh thủ công/điều chỉnh so với prompt khung

- Chuẩn hóa cách biểu diễn Accuracy trong schema sáu cột: đặt tại `f1_score`,
  để `precision`/`recall` là NaN, ghi tổng support và `row_type='summary'`.
- QA bổ sung `row_type` để một class có display name `accuracy`, `macro avg`
  hoặc `weighted avg` không bị visualizer loại nhầm như một summary row.
- Không thêm tham số `class_names` riêng; display name đã được ghép đúng class
  trong `per_class_metrics`, tránh nguy cơ lệch thứ tự ở bước report.
- Không ghi nhận một đợt chỉnh tay độc lập ngoài việc đối chiếu code với kế hoạch.

### Kết quả kiểm thử

- Test trực tiếp nằm trong `tests/test_integration.py`.
- Trạng thái suite integration cuối: 7/7 test passed.
- Test report xác nhận thứ tự `Negative`, `Positive`, `accuracy`, `macro avg`,
  `weighted avg`; support `[5,4,9,9,9]`; Accuracy `7/9` và Precision ở row
  Accuracy là NaN.
- Test QA xác nhận `row_type` đúng và class có tên trùng reserved summary text
  vẫn xuất hiện trong biểu đồ per-class.

### Kết luận

- Module report sẵn sàng tích hợp: Có.

---

## P05 — Visualizations

- Ngày thực hiện: 2026-08-20
- Công cụ AI: OpenAI Codex
- Người thực hiện: Codex
- File đầu ra: `evaluation/visualizations.py`
- Mục tiêu: Vẽ và lưu sáu nhóm biểu đồ từ kết quả đã tính sẵn.

### Prompt đã sử dụng

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

Hãy viết evaluation/visualizations.py bằng Matplotlib Figure/Axes API.

Tạo các hàm public:
1. plot_confusion_matrix
2. plot_overall_metrics_bar
3. plot_per_class_metrics
4. plot_binary_roc_curve
5. plot_multiclass_roc_curves
6. plot_correct_incorrect_pie

Yêu cầu:
- Dùng backend Agg để chạy được trong CI/headless. Mỗi hàm chuẩn hóa output_path,
  tạo thư mục cha, chỉ nhận đuôi .png, lưu bằng fig.savefig với DPI mặc định 300,
  đóng figure bằng plt.close(fig) kể cả khi save lỗi, và trả về pathlib.Path.
- plot_confusion_matrix nhận matrix, class_names, normalized, output_path. Tự
  dùng imshow/colorbar và ax.text để ghi từng ô; count hiện integer, normalized
  hiện hai chữ số thập phân. Không dùng ConfusionMatrixDisplay.
- Vẽ hai file riêng từ runner: confusion_matrix_counts.png và
  confusion_matrix_normalized.png. Module không tự tính normalized matrix.
- plot_overall_metrics_bar chỉ nhận mapping metric đã tính, trục Y [0,1], có
  nhãn giá trị trên bar.
- plot_per_class_metrics nhận report DataFrame có `row_type`, chỉ chọn các dòng
  `row_type='class'` rồi vẽ grouped bars Precision, Recall, F1 cho mỗi lớp,
  Y [0,1]. Không lọc bằng nội dung text của label.
- plot_binary_roc_curve nhận fpr, tpr, auc; plot curve và đường random (0,0)-(1,1).
- plot_multiclass_roc_curves nhận kết quả OVR, class_names, macro_auc; không vẽ
  curve của lớp defined=False, đưa AUC mỗi lớp hợp lệ vào legend và macro AUC
  vào title, thêm random line. Nếu toàn bộ curve undefined, vẫn phải lưu một
  placeholder PNG có đường random và annotation giải thích ROC-AUC undefined.
- plot_correct_incorrect_pie chỉ được phép tính số y_true == y_pred và số sai;
  không tính lại metric classification.
- Kiểm tra ma trận vuông/không âm/hữu hạn, metric/ROC thuộc [0,1], ROC array cùng
  độ dài, class names phù hợp, DPI là integer dương.
- Dùng Pandas duy nhất để nhận report table; Matplotlib duy nhất cho hình.
- Thêm type hint, docstring và ví dụ/test xác nhận file tồn tại, không rỗng và
  không để figure mở sau khi lưu.
```

### Code AI tạo

- Các hàm public: đủ 6 hàm trong prompt.
- Helper nội bộ: validate matrix/class names, metric values, ROC points, output
  path và save/close an toàn.
- Số dòng sau QA cuối: 347.
- Thư viện import: `pathlib`, `typing`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `pandas`.
- Backend: `Agg`; DPI mặc định: 300.

### Kiểm tra điều kiện

- Figure/Axes API, không dùng ConfusionMatrixDisplay: Đạt.
- Tự annotate từng ô Confusion Matrix: Đạt.
- Có raw và normalized qua tham số và hai lần gọi runner: Đạt.
- Bar chart giới hạn Y từ 0 đến 1: Đạt.
- ROC có đường random: Đạt.
- Toàn bộ multiclass ROC undefined vẫn tạo placeholder có giải thích: Đạt.
- Chọn class rows bằng `row_type`, không nhầm class name với summary text: Đạt.
- Pie chỉ tính correct/incorrect: Đạt.
- Lưu PNG và đóng figure: Đạt.
- Không tính lại metric classification: Đạt.

### Điều chỉnh thủ công/điều chỉnh so với prompt khung

- Ép backend `Agg` ngay trước import pyplot để runner hoạt động trong server/CI.
- Gom thao tác lưu/đóng vào `_save_and_close`; dùng `finally` để tránh rò rỉ figure.
- Với multiclass, chỉ plot curve `defined=True`; QA thay lỗi khi không có curve
  hợp lệ bằng placeholder plot rõ trạng thái undefined để pipeline vẫn hoàn tất.
- QA chuyển lọc per-class chart từ so sánh chuỗi reserved sang cột `row_type`.
- Không ghi nhận một đợt chỉnh tay độc lập ngoài việc đối chiếu code với kế hoạch.

### Kết quả kiểm thử

- Hình được kiểm tra end-to-end trong `tests/test_integration.py` cho cả binary
  và multiclass runner.
- Trạng thái suite integration cuối: 7/7 test passed.
- Kiểm tra liên quan: mọi Path trả về tồn tại và có kích thước > 0; binary và
  multiclass tạo đúng ROC PNG; `plt.get_fignums()` rỗng sau khi chạy.
- Case toàn bộ OVR curve undefined xác nhận placeholder ROC vẫn được tạo.

### Kết luận

- Module visualizations sẵn sàng cho prediction-array runner: Có.

---

## P06 — Exporter

- Ngày thực hiện: 2026-08-20
- Công cụ AI: OpenAI Codex
- Người thực hiện: Codex
- File đầu ra: `evaluation/exporters.py`
- Mục tiêu: Xuất bảng đánh giá, prediction và điểm ROC với cấu trúc ổn định.

### Prompt đã sử dụng

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

Hãy viết evaluation/exporters.py với hàm public export_evaluation_results.

Đầu vào gồm metrics, report_df, count/normalized Confusion Matrix, y_true,
y_pred, y_proba, classes, output_dir, roc_results tùy chọn, class_names tùy
chọn và positive_label cho binary.

Yêu cầu:
- Tạo output_dir/tables và output_dir/predictions nếu chưa tồn tại. Báo lỗi nếu
  output_dir đang là file.
- Xuất UTF-8 và index=False:
  1. tables/metrics_summary.csv
  2. tables/classification_report.csv
  3. tables/confusion_matrix_counts.csv
  4. tables/confusion_matrix_normalized.csv
  5. predictions/predictions.csv
  6. tables/roc_points.csv cho binary, hoặc một roc_points_class_<index>_<name>.csv
     cho mọi lớp multiclass, kể cả lớp có ROC undefined.
- metrics_summary chỉ chứa các scalar metric được hỗ trợ theo thứ tự ổn định:
  accuracy, binary precision/recall/f1/roc_auc, macro và weighted metrics,
  roc_auc_macro khi hiện diện. Nếu metric hiện diện nhưng giá trị là None, giữ
  row đó với value trống/NaN để biểu diễn trạng thái undefined.
- Hai Confusion Matrix CSV có cột true_label và các cột
  predicted_<safe_name>. Mọi header sau sanitize phải duy nhất; thêm suffix
  `_2`, `_3`, ... khi nhiều class names chuẩn hóa thành cùng chuỗi.
- predictions luôn có actual_label, predicted_label, probability column(s),
  is_correct. Binary một chiều/hai chiều chỉ xuất xác suất positive_label với
  tên probability_<positive class name>. Multiclass xuất một cột cho mỗi class
  theo đúng thứ tự y_proba; xử lý tên đã sanitize bị trùng bằng suffix.
- ROC CSV defined chứa threshold, fpr, tpr cùng độ dài. Với lớp multiclass
  undefined, vẫn xuất một status row có threshold/fpr/tpr trống,
  defined=False và reason thay vì bỏ mất lớp đó.
- Kiểm tra matrix shape/finite/nonnegative, label vector, probability shape,
  finite/range, class names và positive_label binary trước khi ghi.
- Trả dictionary chứa toàn bộ Path đã tạo; roc_points multiclass là mapping từ
  class label sang Path.
- Không tính lại metric hoặc ROC trong exporter. Pandas chỉ tổ chức bảng và ghi CSV.
- Thêm type hint, docstring, lỗi ValueError cụ thể và integration test đọc lại CSV.
```

### Code AI tạo

- Hàm public: `export_evaluation_results`.
- Helper nội bộ: tạo summary/matrix/prediction DataFrame, xuất ROC, chuẩn bị
  directory, validate shape, lookup class result và sanitize tên.
- Số dòng sau QA cuối: 401.
- Thư viện import: `pathlib`, `typing`, `numpy`, `pandas`.

### Kiểm tra điều kiện

- Tự tạo thư mục output: Đạt.
- CSV UTF-8, index=False: Đạt.
- Đủ metrics/report/two matrices/predictions/ROC points: Đạt.
- Probability columns dùng class name và đúng thứ tự classes: Đạt.
- Confusion/prediction headers sau sanitize luôn duy nhất: Đạt.
- Binary bắt buộc positive_label: Đạt.
- Multiclass ROC xuất riêng từng lớp, gồm status cho lớp undefined: Đạt.
- `roc_auc_macro=None` được xuất thành row có value trống: Đạt.
- Không tính lại metric/ROC: Đạt.
- Trả dictionary Path: Đạt.

### Điều chỉnh thủ công/điều chỉnh so với prompt khung

- Chia output thành `tables/` và `predictions/` để nhất quán với cấu trúc dự án.
- Binary chỉ xuất một cột xác suất của positive class ngay cả khi đầu vào có hai
  cột, nhằm làm rõ semantic của prediction file.
- Tên cột/tệp được sanitize; tên probability trùng được thêm suffix để tránh
  ghi đè cột.
- QA dùng cùng cơ chế unique-name cho cả `predicted_*` headers của Confusion
  Matrix và `probability_*` headers của predictions.
- QA giữ đầy đủ provenance khi ROC undefined: metrics summary có
  `roc_auc_macro` trống và ROC CSV theo lớp có `defined=False` cùng `reason`.
- Bảng tổng hợp/report/matrix dùng sáu chữ số thập phân; prediction scores và
  ROC points dùng 12 chữ số có nghĩa để có thể tái tạo ranking/AUC từ artifact.
- Không ghi nhận một đợt chỉnh tay độc lập ngoài việc đối chiếu code với kế hoạch.

### Kết quả kiểm thử

- Exporter được kiểm tra end-to-end trong `tests/test_integration.py`.
- Trạng thái suite integration cuối: 7/7 test passed.
- Binary xác nhận đúng năm bảng trong `tables/`, `predictions.csv`, tên cột
  `probability_positive`; multiclass xác nhận ba probability columns và ba ROC
  CSV theo lớp. Mọi file được kiểm tra tồn tại và không rỗng.
- QA xác nhận lớp ROC undefined vẫn có CSV status, `roc_auc_macro` để trống,
  header sanitize bị trùng được phân biệt, và binary probability matrix lấy
  đúng cột của `positive_label` kể cả khi positive class nằm ở cột 0.

### Kết luận

- Exporter sẵn sàng dùng với runner độc lập: Có.

---

## P07 — Prediction-array integration runner

- Ngày thực hiện: 2026-08-20
- Công cụ AI: OpenAI Codex
- Người thực hiện: Codex
- File đầu ra hiện tại: `experiments/run_classification_evaluation.py`
- File ghi phần hoãn: `PROJECT_INTEGRATION_NOTES.md`
- Mục tiêu hiện tại: Điều phối toàn bộ pipeline từ các mảng kết quả đã có, không giả định API model/repo.

### Prompt đã sử dụng

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

Hãy viết experiments/run_classification_evaluation.py cho giai đoạn chưa có repo
model/dataset. Không import LightGBM và không nhận model/X_test trong lõi hiện
tại. Thay vào đó, xây dựng prediction-array runner có thể tái sử dụng sau này.

Tạo hàm:
run_classification_evaluation(y_true, y_pred, y_proba, classes, output_dir,
class_names=None, positive_label=None, task_type=None, save_dpi=300) -> dict.

Luồng bắt buộc:
1. Gọi validate_evaluation_inputs để lấy cấu hình/array đã kiểm tra.
2. Gọi evaluate_classification; không viết lại bất kỳ công thức metric nào.
3. Binary: extract positive scores, calculate_binary_roc_curve và thêm roc_auc;
   overall bar dùng Accuracy, positive Precision/Recall/F1 và ROC-AUC.
4. Multiclass: calculate_multiclass_roc_ovr và thêm roc_auc_macro. Cho phép
   macro AUC là None khi mọi lớp thiếu positive/negative; khi đó không đưa Macro
   ROC-AUC vào overall bar nhưng vẫn tiếp tục report, placeholder ROC và export.
5. Gọi create_classification_report_dataframe.
6. Tạo figures directory và gọi đủ: count matrix, normalized matrix, overall
   metrics bar, per-class metrics, correct/incorrect pie và ROC binary hoặc OVR.
7. Gọi export_evaluation_results để xuất toàn bộ CSV/predictions/ROC points.
8. Sau khi toàn bộ bảng/hình của run mới thành công, cập nhật
   output_dir/evaluation_manifest.json và chỉ xóa artifact stale do manifest
   trước đó sở hữu. Không dùng glob; kiểm tra mọi relative path không tuyệt đối,
   không chứa `..` và resolve vẫn nằm trong output_dir. Manifest hỏng chỉ warning
   rồi bỏ cleanup; không xóa file/thư mục ngoài phạm vi hoặc file người dùng.
9. Trả dictionary gồm metadata, metrics, classification_report, roc_results,
   table_paths, figure_paths và manifest_path.

Validate `save_dpi` ngay đầu hàm, trước khi tạo/sửa output: phải là integer dương
và không chấp nhận bool. Nhờ đó một run lỗi validation không cleanup artifact tốt
của lần chạy trước.

Thêm main() và if __name__ == '__main__'. Trong giai đoạn chưa kết nối repo,
main chỉ in hướng dẫn import hàm và trỏ tới PROJECT_INTEGRATION_NOTES.md; không
tạo dữ liệu giả hoặc hard-code dataset/class/positive label/output path. Hỗ trợ
chạy trực tiếp `python experiments/run_classification_evaluation.py`: chỉ khi
`__package__` rỗng mới bootstrap project root vào sys.path; khi import như module
thì không thay đổi sys.path.

Ghi rõ trong module docstring và PROJECT_INTEGRATION_NOTES.md rằng adapter model
bị hoãn. Adapter sau này phải:
- nhận/import model và đúng X_test/y_test từ repo chung;
- xác nhận binary/multiclass, target, ordered classes, class_names, positive_label;
- với LGBMClassifier kiểm tra model.classes_ và predict_proba columns;
- với native Booster, hiểu predict thường trả probability; binary threshold do
  project cung cấp, multiclass dùng argmax rồi map qua classes;
- chuẩn hóa output theo hợp đồng rồi chỉ gọi run_classification_evaluation.

Không tự suy đoán threshold 0.5, không chia lại dataset, không huấn luyện model,
không hard-code đường dẫn. Thêm type hint, docstring và integration test cho cả
binary lẫn multiclass prediction arrays.
```

### Code AI tạo

- Hàm public: `run_classification_evaluation`.
- Entry point: `main` và guard `if __name__ == '__main__'`.
- Helper output lifecycle: cập nhật/đọc manifest, thu thập Path lồng nhau, tạo
  relative path và kiểm tra manifest target an toàn.
- Số dòng sau QA cuối: 361.
- Thư viện import: `json`, `sys`, `warnings`, `pathlib`, `typing` và các function
  từ sáu module `evaluation`.
- Không import `lightgbm`; runner lõi nhận trực tiếp `y_true`, `y_pred`, `y_proba`.

### Kiểm tra điều kiện

- Không viết lại công thức metric/ROC/report/export: Đạt.
- Gọi đúng các module chuyên trách: Đạt.
- Hỗ trợ binary và multiclass từ prediction arrays: Đạt.
- Sinh đủ bảng/hình và trả structured result: Đạt.
- Có `main` và name guard: Đạt.
- Chạy trực tiếp file từ project root bằng bootstrap có điều kiện: Đạt.
- Validate DPI trước khi chạm output: Đạt.
- Không fail khi toàn bộ multiclass ROC undefined: Đạt.
- Manifest cleanup stale path-safe, không glob và chỉ sau run thành công: Đạt.
- Không hard-code dataset/class/positive label/output path: Đạt.
- Nhận model và gọi predict trực tiếp: Hoãn có chủ đích vì repo chính chưa có.

### Điều chỉnh thủ công/điều chỉnh so với prompt khung

- P07 gốc giả định đã có model, `X_test` và API predict. Hiện trạng chuyển ranh
  giới tích hợp sang prediction arrays để không đoán sai API/model/classes.
- `main()` chỉ thông báo cách dùng thay vì chạy demo hard-coded.
- QA bổ sung bootstrap có điều kiện để direct entrypoint không lỗi import
  `evaluation`, đồng thời không sửa sys.path khi dùng package import bình thường.
- QA validate `save_dpi` trước mọi xử lý; multiclass `macro_auc=None` được giữ
  như trạng thái undefined thay vì raise, và visualizer/exporter xử lý tiếp.
- QA thêm `evaluation_manifest.json`: run thành công mới cleanup đúng những file
  stale được manifest cũ liệt kê; không glob, không theo path traversal và không
  đụng file không thuộc pipeline. Run lỗi không thực hiện cleanup.
- Toàn bộ thông tin cần bổ sung cho adapter được note tại
  `PROJECT_INTEGRATION_NOTES.md`, gồm API model, target, classes, class names,
  positive label, native threshold và output directory.
- Đây là phần cố ý chưa hoàn tất, không phải lỗi thiếu implementation của lõi.

### Kết quả kiểm thử

- File kiểm thử: `tests/test_integration.py`.
- Trạng thái suite integration cuối: 7/7 test passed.
- Binary runner xác nhận Accuracy và AUC đều `0.75`, đủ CSV/PNG và prediction
  columns đúng semantic positive class.
- Multiclass runner xác nhận task metadata, macro AUC, ROC PNG OVR, ba ROC CSV
  và probability columns theo đúng class order.
- QA xác nhận all-undefined ROC vẫn hoàn tất output; tái sử dụng cùng output_dir
  xóa đúng artifact pipeline stale nhưng giữ file ngoài manifest; `save_dpi`
  lỗi không cleanup run trước; manifest được trả về và path hợp lệ.
- Binary matrix shape `(n, 2)` với `positive_label` nằm ở cột 0 được kiểm tra để
  xác nhận ROC và `predictions.csv` lấy đúng cột theo thứ tự `classes`.

### Kết luận

- Prediction-array runner sẵn sàng: Có.
- Adapter model/repo chính sẵn sàng: Chưa; chờ người dùng kết nối repo và cung
  cấp các quyết định trong `PROJECT_INTEGRATION_NOTES.md`.

---

## P08 — Unit tests và integration tests

- Ngày thực hiện: 2026-08-20
- Công cụ AI: OpenAI Codex
- Người thực hiện: Codex
- File đầu ra:
  - `tests/test_input_validation.py`
  - `tests/test_manual_metrics.py`
  - `tests/test_manual_roc_auc.py`
  - `tests/test_integration.py`
- Mục tiêu: Kiểm chứng bằng đáp án tính tay và filesystem tạm, không dùng sklearn làm oracle.

### Prompt đã sử dụng

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

Hãy viết test bằng unittest, không dùng sklearn để tạo expected value hoặc đối
chiếu. Dùng NumPy testing cho array, assertAlmostEqual cho float và
TemporaryDirectory cho output.

Tạo bốn file:

1. tests/test_input_validation.py
- Test label array số/chuỗi, giữ thứ tự, non-1D, empty, unequal length, NaN,
  infinity và unsupported label.
- Test Python integer có độ lớn tùy ý để chắc chắn validation không ép float và
  không overflow.
- Test classes empty/duplicate/order và observed label ngoài classes.
- Test class_names mặc định/tùy chỉnh/sai count/blank/non-string.
- Test binary vector, binary two-column matrix, multiclass matrix.
- Test probability ngoài [0,1], NaN/infinity/non-numeric, sai sample count, sai
  số cột, multiclass 1D, row sum sai và row sum nằm trong tolerance.
- Test positive index, thiếu/không thuộc positive label, sai binary class count.
- Test metadata tổng hợp, task inference, task_type không nhất quán và
  `positive_label` bị từ chối khi task là multiclass.

2. tests/test_manual_metrics.py
- Binary fixture y_true=[0,0,1,1,1], y_pred=[0,1,1,0,1], classes=[0,1].
  Expected matrix [[1,1],[1,2]], positive TP=2,TN=1,FP=1,FN=1,
  Accuracy=3/5, Precision=Recall=F1=2/3.
- Test safe_divide scalar/status/custom undefined value và invalid scalar.
- Test string label order, multiclass matrix, normalized empty row, invalid matrix.
- Test per-class fields và macro/weighted average bằng số tính tay.
- Test không dự đoán positive: numeric result 0 và cờ undefined chính xác.
- Test perfect prediction, string binary, multiclass, positive_label bắt buộc và
  cấu hình không hợp lệ.

3. tests/test_manual_roc_auc.py
- Test convert string labels và positive score extraction từ vector/matrix.
- AUC 0.75 cho y_true=[0,0,1,1], score=[0.1,0.4,0.35,0.8].
- Test điểm đầu (0,0), threshold infinity và điểm cuối (1,1).
- Test tất cả score bằng 0.5 tạo một group và AUC 0.5, không phụ thuộc permutation.
- Test perfect AUC 1, thiếu positive/negative, length mismatch, NaN.
- Test trapezoid validation: unequal lengths, decreasing FPR, ngoài [0,1].
- Test multiclass perfect OVR, absent class warning/exclusion, no defined class
  cho macro_auc=None và probability contract sai.

4. tests/test_integration.py
- Dùng matplotlib backend Agg.
- Test report có rows lớp + accuracy + macro avg + weighted avg, đúng support,
  NaN và `row_type`; class name trùng `accuracy`/`macro avg` vẫn được vẽ như lớp.
- Chạy binary prediction-array runner trong TemporaryDirectory; kiểm tra metric
  0.75, AUC 0.75, chính xác danh sách table files, predictions columns, tất cả
  Path tồn tại và file size > 0, không còn figure mở.
- Chạy binary với y_proba shape (n,2), positive_label nằm tại cột 0; kiểm tra
  positive_index, AUC và probability column lấy đúng cột theo classes.
- Chạy multiclass runner; kiểm tra task, macro AUC, một ROC CSV cho mỗi lớp,
  roc_ovr_multiclass.png, unique sanitized probability/confusion headers theo
  class_names và không còn figure.
- Chạy multiclass mà mọi OVR curve undefined; pipeline vẫn thành công, tạo
  placeholder ROC PNG, một status ROC CSV cho mỗi lớp và row roc_auc_macro có
  value trống trong metrics summary.
- Tái sử dụng cùng output_dir qua binary -> multiclass -> binary. Xác nhận
  manifest xóa đúng artifact pipeline stale, giữ file người dùng, và một run
  lỗi `save_dpi` không xóa output tốt từ lần trước.

Mọi test phải độc lập, không ghi artifact cố định vào workspace, không gọi mạng,
không cần LightGBM/model thật. Có if __name__ == '__main__': unittest.main().
```

### Code AI tạo

- `tests/test_input_validation.py`: 332 dòng, 42 test.
- `tests/test_manual_metrics.py`: 258 dòng, 19 test.
- `tests/test_manual_roc_auc.py`: 217 dòng, 15 test.
- `tests/test_integration.py`: 383 dòng, 7 test.
- Tổng số test hiện có: 83.
- Thư viện dùng: `unittest`, `tempfile`, `pathlib`, `numpy`, `pandas`,
  `matplotlib`; import các module dự án cần kiểm tra.

### Kiểm tra điều kiện

- Không dùng sklearn làm oracle: Đạt.
- Expected metric/ROC quan trọng được tính tay: Đạt.
- Có binary, multiclass và string labels: Đạt.
- Có zero denominator, perfect prediction và tied score: Đạt.
- Có invalid probability shape/range/finite/row sum: Đạt.
- Có test CSV/PNG trong thư mục tạm: Đạt.
- Có kiểm tra đóng Matplotlib figures: Đạt.
- Có edge case huge integer và multiclass positive_label: Đạt.
- Có edge case reserved class name, all ROC undefined và unique sanitized headers: Đạt.
- Có binary matrix với positive class ở cột 0: Đạt.
- Có kiểm tra tái sử dụng output/manifest và giữ user-owned file: Đạt.
- Adapter model/repo thật: Chưa test vì chưa tồn tại và nằm ngoài phạm vi hiện tại.

### Điều chỉnh thủ công/điều chỉnh so với prompt khung

- Tách validation thành suite riêng với độ phủ chi tiết hơn danh sách tối thiểu P08.
- Integration test chạy trực tiếp prediction-array runner thay vì mock một API
  model chưa được xác định.
- Dùng `TemporaryDirectory` để không tạo artifact kiểm thử tồn dư trong workspace.
- QA mở rộng integration từ output happy path sang schema collision, undefined
  ROC và lifecycle khi tái sử dụng output directory.
- Không ghi nhận một đợt chỉnh tay độc lập ngoài việc đối chiếu test với code hiện tại.

### Kết quả kiểm thử

- Input validation: 42/42 passed.
- Manual metrics: 19/19 passed.
- Manual ROC-AUC: 15/15 passed.
- Integration: 7/7 passed.
- Tổng trạng thái QA cuối: 83/83 passed.

### Kết luận

- Unit/integration tests cho lõi hiện tại: Sẵn sàng.
- Test adapter với model/dataset thật: Hoãn đến khi repo chung được kết nối.

---

## Yêu cầu đã xử lý khi kết nối repo chính

Các mục sau là checklist đã dùng để tạo P09 và P10 mà không sửa mất lịch sử P07:

1. Đường dẫn chính xác của model/training module và test split.
2. API model thực tế và version LightGBM.
3. Shape/dtype thực tế của `predict` và/hoặc `predict_proba`.
4. Target, task type, ordered `classes`, `class_names`, `positive_label`.
5. Prediction threshold do project xác nhận nếu native binary.
6. Code adapter đã thêm/sửa, điều chỉnh thủ công và lý do.
7. Kết quả chạy unit suite, integration với repo và output thực tế.
8. Kết quả quét source cho import/call bị cấm.

Danh sách quyết định và checklist đầy đủ nằm trong
`classification/evaluation/docs/PROJECT_INTEGRATION_NOTES.md`.

---

## P09 — Khảo sát repo và adapter model thật

- Ngày thực hiện: 2026-08-21
- Công cụ AI: OpenAI Codex
- Người thực hiện: Codex theo yêu cầu của người dùng
- Repository: `https://github.com/thanhdoan252024-hash/machine-learning-group-6.git`
- Snapshot bắt đầu tích hợp: nhánh `LightGBM`, commit `32397d0`
- Mục tiêu: Xác minh model/dataset thật và nối prediction output với core mà
  không viết lại metric.

### Prompt đã sử dụng

```text
Người dùng cung cấp repository GitHub machine-learning-group-6 và yêu cầu đọc,
đề xuất hướng chỉnh sửa để phần đánh giá/trực quan hóa hiện có khớp repo tổng.
Sau khi khảo sát, triển khai adapter cho model classification thật.

Yêu cầu bắt buộc:
- Khảo sát cả default branch và nhánh chứa code thật; không đoán từ README.
- Xác định target, task type, ordered classes, positive label, class names,
  predict/predict_proba shape và semantics cột xác suất từ source/dataset thật.
- Adapter phải lấy classes từ model.classes_, không tính lại metric và chỉ gọi
  `classification.evaluation.runner.run_classification_evaluation`.
- Kiểm tra rõ positive label 1 là Machine failure và cột xác suất tương ứng.
- Thêm test adapter với model NumPy thật: shape (n,2), row sum bằng 1 và predict
  khớp classes_[(proba[:,1] >= threshold)].
- Giữ evaluation core độc lập với model và sklearn.
```

### Code AI tạo

- `classification/evaluation/adapter.py`: 155 dòng.
- `classification/evaluation/tests/test_classification_adapter.py`: 189 dòng,
  5 test.
- Import chính: `numpy`, `pathlib`, `typing`, core runner và model repo trong test.
- API public:
  `evaluate_classification_outputs()` và `evaluate_fitted_classifier()`.

### Kiểm tra điều kiện

- Repo thật được đọc từ nhánh `LightGBM`, không từ `main` chỉ có README: Đạt.
- Model custom binary dùng NumPy, không phải package LightGBM: Đã xác minh.
- Dataset thật 10.000 mẫu; target 0=9.661 và 1=339: Đã xác minh.
- `model.classes_ == [0, 1]`, positive label `1`: Đã xác minh.
- `predict_proba` shape `(n,2)`, tổng hàng bằng 1: Đạt.
- Prediction khớp threshold và cột xác suất thứ hai: Đạt.
- Adapter không chứa công thức metric hoặc import sklearn: Đạt.

### Điều chỉnh thủ công/điều chỉnh so với prompt khung

- Chọn nhánh `LightGBM` làm base vì default `main` không chứa pipeline thật.
- Ánh xạ tên lớp theo giá trị `{0: "Không hỏng máy", 1: "Hỏng máy"}` thay vì
  theo vị trí list âm thầm.
- Adapter dataset-specific được phép cố định positive label đã xác minh; core
  generic vẫn bắt buộc người gọi truyền positive label và không hard-code.
- Output convention hiện tại là `classification/evaluation/outputs/` để code,
  tests, artifacts và docs của phần đánh giá nằm cùng một subtree.

### Kết quả kiểm thử

- Adapter targeted tests: 5/5 passed.
- Full discovery ngay sau khi ghép adapter với 83 core tests: 88/88 passed.

### Kết luận

- Hợp đồng model thật tương thích trực tiếp với prediction-array runner.
- Không cần sửa công thức metric, ROC-AUC, report, visualization hoặc exporter.

---

## P10 — Pipeline tái lập, notebook portable và artifact thật

- Ngày thực hiện: 2026-08-21
- Công cụ AI: OpenAI Codex
- Người thực hiện: Codex theo yêu cầu của người dùng
- Mục tiêu: Hoàn tất tích hợp end-to-end, sinh kết quả thật và chuẩn bị commit.

### Prompt đã sử dụng

```text
Người dùng xác nhận phản hồi của giảng viên: sklearn chỉ áp dụng trong chia dữ
liệu thì được chấp nhận; miễn thuật toán và metrics không dùng trực tiếp thư
viện. Giữ sklearn.model_selection.train_test_split cho split stratified.

Người dùng yêu cầu commit các file CSV và PNG lên GitHub.

Triển khai:
- Tạo branch tích hợp từ LightGBM mới nhất.
- Chuyển notebook từ path D:\ tuyệt đối sang repo-relative path, import package
  portable và thêm cell gọi adapter sau predict/predict_proba.
- Thêm random_state=42 cho model để kết quả tái lập.
- Tạo CLI `classification.evaluation.run_machine_failure_evaluation` dùng cùng
  preprocessing, hyperparameter, split 80/20 stratified và output
  `classification/evaluation/outputs/`.
- Giữ sklearn duy nhất ở train_test_split; cấm sklearn.metrics, model.score,
  metric LightGBM và np.trapz/np.trapezoid.
- Chạy full tests, train model 100 estimator trên dữ liệu thật, kiểm tra toàn bộ
  CSV/PNG/manifest, predictions phải có 2.000 dòng.
- Cập nhật README, integration notes, requirements và prompting log.
```

### Code AI tạo

- `classification/evaluation/run_machine_failure_evaluation.py`: 172 dòng.
- `classification/evaluation/tests/test_machine_failure_pipeline.py`: 47 dòng,
  2 test.
- `classification/classification_metrics.py`: facade 39 dòng, không lặp công thức.
- `classification/__init__.py`, `requirements.txt` và cấu trúc output.
- Cập nhật `classification/machine_failure_prediction.ipynb`: 14 cell, path
  portable, seed model và evaluation cell.
- Cập nhật `classification/lightgbm_classification.py`: bỏ side effect
  `np.set_printoptions()` khỏi `predict_proba()`.
- Thêm/cập nhật `README.md`,
  `classification/evaluation/docs/PROJECT_INTEGRATION_NOTES.md`, `.gitignore` và
  tài liệu kế hoạch trong `classification/evaluation/docs/`.

### Kiểm tra điều kiện

- sklearn chỉ xuất hiện ở bước `train_test_split`: Đạt.
- Model và metrics không import/call implementation thư viện: Đạt.
- Notebook chạy được từ repo root hoặc thư mục classification: Đạt bằng kiểm tra
  code cell/import/path; pipeline CLI là entry point chạy thật.
- Dataset contract: `(10000, 6)` feature matrix, không missing, target đúng: Đạt.
- Split tái lập: train 8.000, test 2.000; test labels 0=1.932, 1=68: Đạt.
- Full suite: 90/90 passed.
- Pipeline thật: exit code 0 sau khi sửa CLI tương thích Windows CP1252.
- Predictions CSV: 2.000 dòng, 4 cột: Đạt.
- Sáu PNG: tồn tại, không rỗng, khoảng 300 DPI và đã xem trực quan: Đạt.
- Normalized confusion matrix hiển thị ba chữ số để không che tỷ lệ lỗi nhỏ: Đạt.
- Manifest liệt kê đúng 12 artifact do pipeline sở hữu: Đạt.
- Chạy lại với cùng seed cho SHA-256 giống nhau ở toàn bộ artifact: Đạt.
- Xác suất 12 chữ số cho phép tính lại AUC từ predictions CSV là
  0,974211423700: Đạt.

### Điều chỉnh thủ công/điều chỉnh so với prompt khung

- Ngoại lệ `sklearn.model_selection` được ghi rõ theo xác nhận mới của giảng
  viên; không sửa lịch sử các prompt P01-P08 của core vốn cấm sklearn hoàn toàn.
- Tạo CLI ngoài notebook để artifact đã commit có thể được tái tạo bằng một lệnh.
- Dùng sáu feature explicit thay vì dựa vào vị trí cột; loại target, ID và năm cờ
  failure mode để tránh leakage giống notebook.
- Đặt `random_state=42` cho cả split và model, trong khi notebook cũ chỉ đặt cho
  split.
- Dòng print đầu của CLI được chuyển sang ASCII sau khi run đầu tiên tạo artifact
  nhưng gặp `UnicodeEncodeError` ở console CP1252; run lại thành công.
- Xóa toàn bộ output/execution count cũ trong notebook sau khi phát hiện xác suất
  đã lưu thuộc run trước seed mới; CSV/PNG là nguồn kết quả baseline chuẩn.
- Notebook và CLI dùng chung allowlist `FEATURE_COLUMNS`/`TARGET_COLUMN` để CSV
  có thêm cột trong tương lai không làm hai entry point chọn feature khác nhau.
- Tăng precision khi xuất score và annotation normalized matrix sau QA để giữ
  khả năng tái tạo AUC và không hiển thị false-positive rate nhỏ thành `0.00`.
- Artifact CSV/PNG được commit theo yêu cầu người dùng, không thêm vào `.gitignore`.

### Kết quả kiểm thử

- Full unittest discovery cuối: 90/90 passed.
- Pipeline thật trên 2.000 mẫu test:
  Accuracy=0,986000; Precision=0,900000; Recall=0,661765;
  F1-score=0,762712; ROC-AUC=0,974211.
- Confusion matrix: TN=1.927, FP=5, FN=23, TP=45.
- Output: 5 table CSV, 1 predictions CSV, 6 PNG và 1 manifest.

### Kết luận

- Pipeline classification end-to-end và artifact thật: Đã commit tại `cd3913a`
  và push lên nhánh `classification-evaluation-integration`.
- Tài liệu và lệnh tái tạo kết quả: Đã bổ sung.

---

## P11 — Gom toàn bộ phần evaluation vào package classification

- Ngày thực hiện: 2026-08-21
- Công cụ AI: OpenAI Codex
- Người thực hiện: Codex theo yêu cầu của người dùng
- Mục tiêu: Tái cấu trúc để module, tests, outputs và docs của evaluation nằm
  cùng một subtree, không thay đổi công thức hay baseline.

### Prompt đã sử dụng

```text
Gom toàn bộ phần đánh giá classification vào `classification/evaluation/`.

Yêu cầu bắt buộc:
- Core modules nằm trực tiếp trong package `classification/evaluation/`.
- Đổi runner thành `classification/evaluation/runner.py`, adapter thành
  `classification/evaluation/adapter.py` và CLI thành
  `classification/evaluation/run_machine_failure_evaluation.py`.
- Chuyển tests vào `classification/evaluation/tests/`, artifacts vào
  `classification/evaluation/outputs/` và ba tài liệu vào
  `classification/evaluation/docs/`.
- Cập nhật toàn bộ internal imports, notebook, README, lệnh test và CLI theo
  package path mới.
- Giữ `requirements.txt` và `.gitignore` ở repository root.
- Không sửa thuật toán model, công thức metric, nội dung artifact hay hợp đồng
  public ngoài việc đổi module path.
- Chạy toàn bộ test suite và kiểm tra không còn import/path runtime cũ.
```

### Code AI tạo

- Package hiện tại gồm `classification/evaluation/{adapter.py,runner.py,
  run_machine_failure_evaluation.py,input_validation.py,manual_metrics.py,
  manual_roc_auc.py,reports.py,visualizations.py,exporters.py}`.
- Sáu file test nằm trong `classification/evaluation/tests/`.
- Mười ba artifact baseline nằm trong `classification/evaluation/outputs/`.
- Kế hoạch, integration notes và prompting log nằm trong
  `classification/evaluation/docs/`.
- Import public hiện tại:
  `classification.evaluation.adapter.evaluate_classification_outputs`,
  `classification.evaluation.adapter.evaluate_fitted_classifier` và
  `classification.evaluation.runner.run_classification_evaluation`.

### Kiểm tra điều kiện

- Import nội bộ và import public dùng namespace `classification.evaluation`: Đạt.
- CLI chuẩn là
  `python -m classification.evaluation.run_machine_failure_evaluation`: Đạt.
- Test discovery dùng `classification/evaluation/tests`: Đạt.
- Output mặc định dùng `classification/evaluation/outputs/`: Đạt.
- `requirements.txt` và `.gitignore` vẫn ở repository root: Đạt.
- Không thay đổi model, công thức metric, notebook output hoặc artifact data: Đạt.

### Điều chỉnh thủ công/điều chỉnh so với prompt khung

- Giữ nguyên path lịch sử trong P01-P08 để log phản ánh đúng trình tự tạo core;
  bổ sung ghi chú đầu log để phân biệt với layout hiện tại.
- Chuẩn hóa phần trạng thái hiện tại, P09 và P10 sang path mới; P11 ghi riêng
  thao tác tái cấu trúc để không làm mất provenance của baseline đã commit.
- README root chỉ giữ vai trò entry point; tài liệu chi tiết được đặt cạnh package
  evaluation.

### Kết quả kiểm thử

- Lệnh:
  `python -m unittest discover -s classification/evaluation/tests -p "test_*.py" -v`.
- Full discovery sau khi đổi layout: 90/90 passed.
- Các import public mới và notebook JSON được kiểm tra thành công.
- Cây artifact vẫn gồm 5 table CSV, 1 predictions CSV, 6 PNG và 1 manifest.

### Kết luận

- Layout `classification/evaluation/{modules,tests,outputs,docs}` đã hoàn tất.
- Tài liệu, import path, CLI và lệnh test hiện thống nhất với filesystem.
- Thay đổi P11 sẵn sàng để review và commit; chưa commit trong lượt tái cấu trúc này.
