# Machine Learning Group 6 — LightGBM From Scratch

Repository triển khai LightGBM-style bằng Python/NumPy cho hai bài toán **Regression** và **Binary Classification**. Project không chỉ huấn luyện mô hình mà còn kiểm chứng correctness của core, kiểm soát data leakage, dùng train/validation/test độc lập, thực hiện controlled hyperparameter study, ablation, EFB sparse microbenchmark và so sánh với các mô hình tham chiếu.

> **Quan trọng — số liệu dùng để chấm / báo cáo cuối:** dùng **Phase 3 Final**, không dùng các metric baseline 80/20 ở phần lịch sử phía dưới.

## 1. FINAL SUBMISSION — nguồn số liệu chính

Các nguồn canonical cho kết quả cuối:

- `PHASE3_RESULTS.md` — tóm tắt Phase 3 ở repository root.
- `case_study/PHASE3_RESULTS.md` — bản cùng nội dung trong workspace case study.
- `case_study/artifacts/phase3/phase3_summary.json` — summary machine-readable của toàn Phase 3.
- `case_study/artifacts/phase3/regression_final.json` — cấu hình và metric Regression final.
- `case_study/artifacts/phase3/classification_final.json` — cấu hình, threshold và metric Classification final.
- `case_study/report/Case_Study_Thuc_nghiem_LightGBM_Regression_Classification.docx` — báo cáo Word final.

### 1.1. Regression Final — Pre-exam Student Score Prediction

Protocol: **70/15/15 train/validation/test**. Preprocessing được fit trên train; hyperparameter và số boosting round được chọn bằng validation; test chỉ dùng cho báo cáo cuối và reference comparison đã định nghĩa trước. Các biến hậu kỳ thi/target-derived được loại khỏi primary scenario để tránh leakage.

Cấu hình final:

- `learning_rate = 0.08`
- `num_leaves = 15`
- `max_bins = 32`
- `min_data_in_leaf = 30`
- `reg_lambda = 1.0`
- `feature_fraction = 0.8`
- GOSS = ON
- EFB = ON (không kích hoạt trên dataset dense này)
- boosting budget = 240; early stopping chọn **125 trees**

Test metrics final:

| Metric | Test |
|---|---:|
| MAE | **7.3425** |
| RMSE | **9.2096** |
| R² | **0.6566** |

### 1.2. Classification Final — Machine Failure Prediction

Dataset có 10.000 mẫu, positive rate **3,39%**. Protocol final là **stratified 70/15/15** tương ứng **7.000 / 1.500 / 1.500** mẫu. Hyperparameter, early stopping và decision threshold đều được chọn trên validation; test không dùng để tuning.

Cấu hình final:

- `learning_rate = 0.05`
- `num_leaves = 15`
- `max_depth = 5`
- `max_bins = 127`
- `min_child_samples = 20`
- `reg_lambda = 1.0`
- `feature_fraction = 1.0`
- `scale_pos_weight = 1.0`
- GOSS = ON
- EFB = ON (không kích hoạt trên dataset dense này)
- boosting budget = 450; early stopping chọn **195 trees**
- decision threshold = **0.22**, chọn **trên validation only**

Test metrics final:

| Metric | Test |
|---|---:|
| Accuracy | 0.9753 |
| Precision | 0.6129 |
| Recall | **0.7451** |
| F1 | **0.6726** |
| Balanced Accuracy | **0.8643** |
| ROC-AUC | **0.9637** |
| PR-AUC / Average Precision | **0.7825** |
| False Negatives | **13** |

**PR-AUC, Recall, F1 và Balanced Accuracy được ưu tiên hơn Accuracy** vì positive class chỉ chiếm 3,39%.

## 2. Phạm vi sử dụng thư viện — được phép / không dùng cho core

Bảng dưới đây là **quy ước chính thức của submission**. Mục tiêu của project là tự cài đặt LightGBM-style core; vì vậy thư viện ngoài chỉ được dùng cho I/O, chia dữ liệu, trực quan hóa, kiểm thử và benchmark tham chiếu.

| Thư viện / công cụ | Dùng trong LightGBM From Scratch core? | Vai trò trong package | Trạng thái |
|---|---|---|---|
| Python standard library | Có, cho hạ tầng | file/path, dataclass, timing, JSON | **Được phép** |
| NumPy | Có | mảng số, vectorization, histogram primitives, objective calculations | **Được phép**; không gọi model ML có sẵn |
| pandas | Ngoài core tree | đọc dữ liệu, DataFrame, export artifact | **Được phép** |
| `sklearn.model_selection.train_test_split` | Không | chỉ chia train/validation/test | **Được phép cho split** |
| Các model/transformer khác của scikit-learn | **Không** | chỉ dùng ở **reference comparison**: Dummy, Random Forest, Logistic Regression, StandardScaler | **Cấm dùng để tạo model from-scratch**; benchmark only |
| `lightgbm` official | **Không** | benchmark tham chiếu cuối Phase 3 | **Cấm dùng trong model from-scratch**; reference only |
| matplotlib / seaborn | Không | trực quan hóa kết quả | **Được phép** |
| pytest | Không | automated tests | **Được phép** |

Model Regression/Classification from-scratch, gradient/Hessian, histogram binning, split search, leaf-wise tree, GOSS, EFB, regularization, early stopping logic và các metric chính của case study **không gọi estimator của scikit-learn hay package `lightgbm`**.

## 3. Protocol final và các làm rõ quan trọng

### 3.1. Phase 1 baseline khác Final Phase 3

| Nội dung | Phase 1 baseline — lịch sử | Final Phase 3 — **dùng để chấm** |
|---|---|---|
| Classification split | stratified **80/20** | stratified **70/15/15** |
| Validation riêng | Không | Có |
| Estimator budget / best iteration | 100 estimators cố định | budget 450; **best iteration 195** |
| Decision threshold | **0.50** cố định | **0.22**, chọn trên validation |
| Metric nổi bật | Accuracy **0.9875** | PR-AUC **0.7825**, Recall **0.7451**, F1 **0.6726** |
| Vai trò | Historical / backward-compatibility | **Canonical final submission** |

Phase-1 executable pipeline và output cũ đã được chuyển khỏi nhánh final chính sang:

```text
legacy/phase1_classification/
```

Pipeline final duy nhất nằm tại `classification/case_study_pipeline.py`.

### 3.2. Regression hyperparameter screening dùng subset, sau đó retrain full

Do implementation Python from-scratch có chi phí cao khi lặp nhiều cấu hình, controlled hyperparameter screening của Regression dùng **subset cố định 5.000 train + 2.000 validation**, lấy **chỉ từ split train/validation**. Test không được đọc trong bước chọn cấu hình.

Sau khi freeze cấu hình, model được huấn luyện lại trên **đầy đủ 70.000 train**, early stopping trên **đầy đủ 15.000 validation**, rồi mới báo cáo trên **15.000 test**. Classification có dataset nhỏ hơn nên controlled study dùng toàn bộ train/validation tương ứng.

### 3.3. Missing-value capability khác với protocol metric 100k

Core Regression có **native missing bin / missing branch** và capability này được khóa bằng unit tests (ví dụ NaN được đưa vào missing bin và prediction đi theo missing branch). Tuy nhiên, **primary Regression case study 100.000 dòng không dùng native missing trực tiếp để tạo metric final**: `TrainOnlyPreprocessor` fit **mean trên numeric và mode trên categorical chỉ từ train**, rồi dùng các thống kê đó để transform validation/test.

Vì vậy cần diễn giải đúng:

- **Unit tests** chứng minh implementation có khả năng xử lý missing-bin.
- **Final Regression metrics** (RMSE 9.2096, R² 0.6566) được đo trên dữ liệu đã **train-only mean/mode impute**.
- Classification dataset final không có missing value ở sáu feature được sử dụng.

### 3.4. Official LightGBM benchmark không phải strict iso-iteration budget

Reference benchmark dùng cùng train/validation/test split và các hyperparameter có thể ánh xạ tương ứng, nhưng **không ép hai implementation có đúng cùng số boosting iterations tối đa**. Official LightGBM được cấp `n_estimators=2000` và `early_stopping=50`; mục tiêu là lấy model reference được validation chọn, không phải benchmark strict equal-iteration.

Trong run đã báo cáo:

- Regression: scratch cap **240**, best iteration **125**; Official best iteration **151**.
- Classification: scratch cap **450**, best iteration **195**; Official best iteration **230**.

Cả hai best iteration của Official đều **nằm dưới cap của scratch tương ứng**, nên budget 2000 không làm Official thực tế vượt trần scratch trong run này. Tuy nhiên, so sánh **training time không phải iso-compute benchmark**; nó chủ yếu minh họa khoảng cách giữa Python from-scratch và implementation native tối ưu. So sánh chất lượng dự đoán (RMSE/PR-AUC) là mục tiêu chính của reference comparison.

## 4. Experimental integrity

Phase 3 tuân theo protocol:

```text
Raw data
  -> Train / Validation / Test
  -> preprocessing fit ONLY on train
  -> model fit on train
  -> hyperparameter / early stopping / threshold selected on validation
  -> final test reporting once configuration is frozen
  -> pre-specified reference comparison
```

Official `lightgbm` **không được dùng để huấn luyện model from-scratch**. Package chính thức chỉ xuất hiện ở bước benchmark tham chiếu cuối.

Correctness/final test status:

```text
173 tests passed
2 subtests passed
0 failed
```

Các kiểm chứng chính gồm gradient/Hessian, histogram, EFB equivalence, GOSS, split/tree behavior, early stopping và case-study pipeline.

## 5. Phase 3 evidence

### GOSS ablation

- Regression: tắt GOSS làm validation RMSE xấu hơn và training chậm hơn trong screening.
- Classification: PR-AUC validation giảm từ khoảng **0.7807** xuống **0.7532** khi tắt GOSS.

### EFB

Hai dataset chính khá dense nên EFB không bundle feature thật sự:

- Regression: 37 features -> 37 bundles.
- Classification: 6 features -> 6 bundles.

Để kiểm chứng EFB trong điều kiện phù hợp, sparse microbenchmark riêng cho kết quả:

- 40 mutually-exclusive sparse features -> **1 bundle**.
- Max histogram absolute error ≈ **3.64e-11**.
- Median histogram construction speedup ≈ **4.27x**.

### Reference comparison

Regression test RMSE:

| Model | RMSE |
|---|---:|
| LightGBM From Scratch | **9.20960** |
| Official LightGBM 4.6.0 | **9.20946** |
| Random Forest | 9.51932 |
| Dummy Mean | 15.71611 |

Classification test PR-AUC:

| Model | PR-AUC |
|---|---:|
| Official LightGBM 4.6.0 | **0.8001** |
| LightGBM From Scratch | **0.7825** |
| Random Forest | 0.7763 |
| Logistic Regression | 0.4345 |

Bản from-scratch có chất lượng dự đoán hợp lý và gần Official LightGBM; hạn chế chính là tốc độ Python thuần chậm hơn implementation native tối ưu.

## 6. Cấu trúc chính

```text
lightgbm_from_scratch/
  core/
    binning.py
    histogram.py
    split.py
    tree.py
    goss.py
    efb.py
    regularization.py
  objectives/
    regression.py
    binary.py

regression/
  lightgbm_regression.py
  case_study_pipeline.py
  data/
  tests/
  visualization/

classification/
  lightgbm_classification.py
  case_study_pipeline.py
  data/raw/machine_fail.csv
  data.py                  # protocol-neutral dataset loader
  evaluation/              # reusable manual metrics / ROC / reports

legacy/
  phase1_classification/     # historical 80/20 runner + notebook + outputs

case_study/
  experiments/
  validation/
  visualization/
  artifacts/
    phase3/
      phase3_summary.json
      regression_final.json
      classification_final.json
      *_ablation.csv
      *_hyperparameter_study.csv
      *_reference_comparison.csv
      figures/
      final_figures/
  report/
    Case_Study_Thuc_nghiem_LightGBM_Regression_Classification.docx

PHASE3_RESULTS.md
CHANGELOG_PHASE1.md
CHANGELOG_PHASE2.md
CHANGELOG_PHASE3.md
requirements.txt
```

## 7. Cài đặt

Yêu cầu Python 3.10 trở lên. Từ repository root:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Trên macOS/Linux, dùng `./venv/bin/python` thay cho đường dẫn Python của Windows.

## 8. Chạy kiểm thử

Chạy full discovery từ repository root:

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

Trạng thái final đã ghi trong `case_study/artifacts/phase3/phase3_summary.json`:

```text
173 passed + 2 subtests
0 failed
```

Core correctness validation nằm tại:

```text
case_study/validation/run_core_validation.py
case_study/artifacts/core_validation.json
```

## 9. Pipeline và artifact final

Các pipeline case study chính:

- `regression/case_study_pipeline.py`
- `classification/case_study_pipeline.py`

Các script experiment Phase 3:

- `case_study/experiments/run_phase3_studies.py`
- `case_study/experiments/run_regression_screen_config.py`
- `case_study/experiments/run_regression_ablation_config.py`
- `case_study/experiments/run_classification_screen_config.py`
- `case_study/experiments/run_classification_ablation_config.py`

Các artifact **Phase 3** là artifact final dùng cho báo cáo/chấm. Không lấy metric final từ `legacy/phase1_classification/outputs/`; đây chỉ là output lịch sử của Phase 1.

---

## 10. Phase 1 Baseline — HISTORICAL REFERENCE ONLY

> **Không dùng các số trong mục này làm kết quả final để chấm.** Đây là baseline lịch sử của classification trước khi project chuyển sang protocol 70/15/15, validation-only model selection, early stopping và threshold tuning.

Baseline Phase 1 dùng:

- split **80/20 train/test**;
- `random_state=42`, stratify theo target;
- **100 estimators**;
- threshold cố định **0.5**;
- không có validation split để chọn threshold/model như Phase 3.

Baseline test metrics:

| Metric | Phase 1 baseline test |
|---|---:|
| Accuracy | **0.987500** |
| Precision — máy hỏng | 0.905660 |
| Recall — máy hỏng | 0.705882 |
| F1 — máy hỏng | 0.793388 |
| ROC-AUC | 0.974874 |

Train có 8.000 mẫu và test có 2.000 mẫu. Các artifact tương ứng nằm chủ yếu trong:

```text
legacy/phase1_classification/outputs/
```

Pipeline baseline có thể tái tạo bằng:

```powershell
.\venv\Scripts\python.exe -m legacy.phase1_classification.run_machine_failure_evaluation
```

Các metric baseline này vẫn được giữ để theo dõi lịch sử phát triển và kiểm tra backward compatibility, **không thay thế kết quả Final Phase 3**.

## 11. Quy tắc đọc kết quả trong package

Nếu cần xác định số nào là số cuối cùng, dùng thứ tự ưu tiên sau:

1. `case_study/artifacts/phase3/phase3_summary.json`
2. `case_study/artifacts/phase3/regression_final.json` hoặc `classification_final.json`
3. `PHASE3_RESULTS.md`
4. Báo cáo Word trong `case_study/report/`

Các file baseline/Phase 1 chỉ dùng làm lịch sử và đối chiếu, không dùng làm final submission metrics.
