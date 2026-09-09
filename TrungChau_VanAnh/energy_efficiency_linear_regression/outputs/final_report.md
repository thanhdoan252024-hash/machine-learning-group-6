# Final Report: Energy Efficiency Linear Regression

## Objective
Predict the building Heating Load (Y1) from the architectural variables X1 through X8 using a custom Linear Regression from scratch implemented with NumPy and Gradient Descent, while preventing target leakage by excluding Y2 from the feature matrix used for Y1 training.

## Dataset
The source dataset is the available file in the project data directory. The dataset contains the architectural variables X1..X8 and two target-like variables Y1 (Heating Load) and Y2 (Cooling Load). The custom model is implemented with NumPy and gradient descent only; no sklearn `LinearRegression` is used. Y2 remains out of X to avoid target leakage, and the report focuses on the Y1 heating-load prediction task.

## Metrics

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Mean Baseline | 9.272103 | 10.237590 | -0.005525 |
| Linear Regression | 2.209380 | 3.072657 | 0.909421 |

## Interpretation
The scratch linear model improves substantially over the mean baseline in MAE, RMSE and R2, confirming that the selected feature columns carry meaningful signal for Y1 prediction.
The coefficients are stored in the coefficient table and are sorted by absolute coefficient magnitude to highlight the strongest feature effects.

## Top coefficients

| feature | coefficient |
|---|---:|
| X5 | 6.667911 |
| X8_2 | 3.765914 |
| X8_5 | 3.542273 |
| X8_3 | 3.535259 |
| X8_1 | 3.502795 |

## Chương 6. Prompting Logs và Quy trình xây dựng Source Code
### 6.1. Tổng quan quá trình phát triển mã nguồn
Hành trình chuyển hóa từ công thức toán học sang mã lệnh: từ công thức hồi quy tuyến tính, định nghĩa hàm mất mát và Gradient Descent, đến việc triển khai dự đoán và tối ưu hóa trọng số bằng Python và NumPy, rồi tiến tới kiểm tra bằng unit test và tạo báo cáo phân tích. Quy trình này cần phản ánh rõ: khởi đầu từ lý thuyết toán học, sau đó được cấu trúc thành source code, đánh giá kết quả và chốt sản phẩm.

### 6.2. Quy trình làm việc chuẩn với AI
Xác định yêu cầu → Soạn Prompt có cấu trúc → Nhận Output → Đọc hiểu & Kiểm tra công thức → Chạy thử nghiệm (Unit test) → Refactor & Chốt code.

### 6.3. Bộ nguyên tắc thiết kế Prompt hiệu quả
Rõ ràng mục tiêu, cung cấp ngữ cảnh toán học, ràng buộc đầu vào/đầu ra, giới hạn thư viện (chỉ dùng NumPy), yêu cầu giải thích luồng xử lý, và yêu cầu kiểm tra dữ liệu loại bỏ Y2 để tránh rò rỉ mục tiêu. Prompt nên mô tả đúng chức năng xử lý, không tạo dữ liệu giả lập, và chỉ cho phép mô hình tối ưu hóa trên tập huấn luyện.

### 6.4. Nhật ký Prompting (Prompting Logs) – Linear Regression
Câu lệnh prompt và source code tương ứng trong dự án như sau:

| Câu lệnh prompt | Tác nhân | Source code/khối xử lý |
|---|---|---|
| Tạo hàm dự đoán y = w^T x + b | Linear Regression | `predict()` / `forward()` trong `linear_regression.py` |
| Cập nhật trọng số bằng Gradient Descent | Linear Regression | `fit()` và `update_weights()` trong `linear_regression.py` |
| Tạo mô hình baseline và so sánh với mô hình học được | Mean Baseline | `mean_baseline()` trong `metrics.py` |
| Tách dữ liệu huấn luyện và kiểm thử | Preprocessing | `custom_train_test_split()` trong `preprocessing.py` |
| Chuẩn hóa và mã hóa đặc trưng | Preprocessing | `fit_numeric_medians()`, `apply_numeric_medians()`, `onehot_encode_train_test()` trong `preprocessing.py` |
| Đánh giá mô hình bằng MAE/RMSE/R2 | Metrics | `mae()`, `rmse()`, `r2_score()` trong `metrics.py` |
| Trực quan hóa trực tiếp các điều cần kiểm tra | Visualization | `plot_training_loss()`, `plot_actual_vs_predicted()`, `plot_residuals()`, `plot_residual_distribution()` trong `visualization.py` |

### 6.5. Tiền xử lý và ràng buộc sớm
Dữ liệu được đọc trực tiếp từ `energy_efficiency_building_heating_cooling_load_dataset.csv`. Trong quá trình preprocessing, Y2 được loại bỏ khỏi khung đặc trưng X để tránh hiện tượng data leakage. Tập train/test được tách trước khi imputation, one-hot và scaling áp dụng trên tập train; sau đó, các phép biến đổi được áp dụng thống nhất lên tập test. Đây là một quy trình an toàn dạng train-only fit và test-only transform, và chỉ mục tiêu Y1 được tối ưu hóa bởi mô hình.

### 6.8. Tiêu chí đánh giá chất lượng Prompting Logs
Độ chuẩn xác kỹ thuật của prompt, khả năng kiểm soát kết quả, số lần phản hồi điều chỉnh, mức độ thấu hiểu code của người lập trình, đồng thời khả năng truy trace từ biểu thức toán học đến source code. Một prompt chất lượng tốt phải giúp con người kiểm tra mối nối giữa công thức học máy và mã nguồn được mô tả rõ ràng, có khả năng tái chạy được và chịu được thử nghiệm unit. Trong dự án này, chất lượng log được xác định qua việc nguồn dữ liệu rõ, mục tiêu Y1 rõ, Y2 bị loại ‘không cho vào X’, và mô hình custom NumPy được kiểm chứng bằng test.

### 6.9. Cấu trúc Source Code thuật toán – Linear Regression
Cấu trúc mã nguồn của thuật toán Linear Regression từ scratch theo đúng quy trình học máy được tổ chức thành các module chính: `data_loader.py` đọc dữ liệu; `eda.py` và `stage2_cleaning_eda.py` kiểm tra dữ liệu và EDA; `preprocessing.py` thực hiện train/test split, imputation, one-hot, và chuẩn hóa; `linear_regression.py` chứa lớp `LinearRegressionScratch` dùng Gradient Descent để cập nhật trọng số và bias; `metrics.py` chứa MAE, RMSE, R2 và baseline; `visualization.py` sinh các plot quan sát; `stage5_training_eval.py` thực hiện huấn luyện và đánh giá; `stage6_report.py` sinh báo cáo giải thích tham số; `stage7_final_artifacts.py` đóng gói văn bản và notebook cuối cùng.

### 6.10. Bảng đối chiếu lý thuyết toán học và hàm mã nguồn tương ứng

| Nội dung toán học cốt lõi | Thuật toán | Tên hàm / khối xử lý trong source code |
|---|---|---|
| Dự đoán y = w^T x + b | Linear Regression | `predict()` / `forward()` trong `linear_regression.py` |
| Cập nhật trọng số bằng Gradient Descent | Gradient Descent | `fit()` trong `LinearRegressionScratch` và `loss_history` |
| Tính ma trận hệ số và bias | Linear algebra | `get_coefficients()` và `get_intercept()` trong `linear_regression.py` |
| Tách train/test trước xử lý | Data splitting | `custom_train_test_split()` trong `preprocessing.py` |
| Chuyển đổi biến phân loại | One-hot encoding | `onehot_encode_train_test()` trong `preprocessing.py` |
| Điền thiếu bằng trung vị train-only | Imputation | `fit_numeric_medians()` và `apply_numeric_medians()` trong `preprocessing.py` |
| Chuẩn hóa đặc trưng bằng train mean/std | Scaling | `preprocess_pipeline()` trong `preprocessing.py` |
| Đánh giá hiệu quả mô hình | MAE/RMSE/R2 | `mae()`, `rmse()`, `r2_score()` trong `metrics.py` |
| So sánh với baseline mean | Mean Baseline | `mean_baseline()` trong `metrics.py` |
| Curve học loss | Optimization trace | `plot_training_loss()` trong `visualization.py` |
| Đồ thị thực tế và dự đoán | Residual diagnostics | `plot_actual_vs_predicted()` và `plot_residuals()` trong `visualization.py` |

## Outputs
- Figures saved under `outputs/figures/`
- Metrics report saved under `outputs/tables/metrics_report.txt`
- Coefficient table saved under `outputs/tables/model_coefficients.csv`
- Narrative report saved under `outputs/stage6_report.txt`
