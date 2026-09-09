import sys
from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'energy_efficiency_building_heating_cooling_load_dataset.csv'
OUT_DIR = PROJECT_ROOT / 'outputs'
FIG_DIR = OUT_DIR / 'figures'
TABLE_DIR = OUT_DIR / 'tables'
NOTEBOOK_DIR = PROJECT_ROOT / 'notebooks'


def parse_metrics_report(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            if not line.strip():
                continue
            model, mae, rmse, r2 = line.strip().split(',')
            rows.append({
                'Model': model,
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R2': float(r2),
            })
    return pd.DataFrame(rows)


def parse_coefficients(path):
    df = pd.read_csv(path)
    return df.sort_values('absolute_coefficient', ascending=False)


def generate_final_markdown(metrics_df, coeff_df):
    # Format a short final report in markdown.
    metrics_text = []
    for _, row in metrics_df.iterrows():
        metrics_text.append(
            f"| {row['Model']} | {row['MAE']:.6f} | {row['RMSE']:.6f} | {row['R2']:.6f} |"
        )

    top_features = coeff_df.head(5)
    lines = []
    lines.append('# Final Report: Energy Efficiency Linear Regression')
    lines.append('')
    lines.append('## Objective')
    lines.append('Predict the building Heating Load (Y1) from the architectural variables X1 through X8 using a custom Linear Regression from scratch implemented with NumPy and Gradient Descent, while preventing target leakage by excluding Y2 from the feature matrix used for Y1 training.')
    lines.append('')
    lines.append('## Dataset')
    lines.append('The source dataset is the available file in the project data directory. The dataset contains the architectural variables X1..X8 and two target-like variables Y1 (Heating Load) and Y2 (Cooling Load). The custom model is implemented with NumPy and gradient descent only; no sklearn `LinearRegression` is used. Y2 remains out of X to avoid target leakage, and the report focuses on the Y1 heating-load prediction task.')
    lines.append('')
    lines.append('## Metrics')
    lines.append('')
    lines.append('| Model | MAE | RMSE | R2 |')
    lines.append('|---|---:|---:|---:|')
    lines.extend([f'| {row["Model"]} | {row["MAE"]:.6f} | {row["RMSE"]:.6f} | {row["R2"]:.6f} |' for _, row in metrics_df.iterrows()])
    lines.append('')
    lines.append('## Interpretation')
    lines.append('The scratch linear model improves substantially over the mean baseline in MAE, RMSE and R2, confirming that the selected feature columns carry meaningful signal for Y1 prediction.')
    lines.append('The coefficients are stored in the coefficient table and are sorted by absolute coefficient magnitude to highlight the strongest feature effects.')
    lines.append('')
    lines.append('## Top coefficients')
    lines.append('')
    lines.append('| feature | coefficient |')
    lines.append('|---|---:|')
    for _, row in top_features.iterrows():
        lines.append(f"| {row['feature']} | {row['coefficient']:.6f} |")
    lines.append('')
    lines.append('## Chương 6. Prompting Logs và Quy trình xây dựng Source Code')
    lines.append('### 6.1. Tổng quan quá trình phát triển mã nguồn')
    lines.append('Hành trình chuyển hóa từ công thức toán học sang mã lệnh: từ công thức hồi quy tuyến tính, định nghĩa hàm mất mát và Gradient Descent, đến việc triển khai dự đoán và tối ưu hóa trọng số bằng Python và NumPy, rồi tiến tới kiểm tra bằng unit test và tạo báo cáo phân tích. Quy trình này cần phản ánh rõ: khởi đầu từ lý thuyết toán học, sau đó được cấu trúc thành source code, đánh giá kết quả và chốt sản phẩm.')
    lines.append('')
    lines.append('### 6.2. Quy trình làm việc chuẩn với AI')
    lines.append('Xác định yêu cầu → Soạn Prompt có cấu trúc → Nhận Output → Đọc hiểu & Kiểm tra công thức → Chạy thử nghiệm (Unit test) → Refactor & Chốt code.')
    lines.append('')
    lines.append('### 6.3. Bộ nguyên tắc thiết kế Prompt hiệu quả')
    lines.append('Rõ ràng mục tiêu, cung cấp ngữ cảnh toán học, ràng buộc đầu vào/đầu ra, giới hạn thư viện (chỉ dùng NumPy), yêu cầu giải thích luồng xử lý, và yêu cầu kiểm tra dữ liệu loại bỏ Y2 để tránh rò rỉ mục tiêu. Prompt nên mô tả đúng chức năng xử lý, không tạo dữ liệu giả lập, và chỉ cho phép mô hình tối ưu hóa trên tập huấn luyện.')
    lines.append('')
    lines.append('### 6.4. Nhật ký Prompting (Prompting Logs) – Linear Regression')
    lines.append('Câu lệnh prompt và source code tương ứng trong dự án như sau:')
    lines.append('')
    lines.append('| Câu lệnh prompt | Tác nhân | Source code/khối xử lý |')
    lines.append('|---|---|---|')
    lines.append('| Tạo hàm dự đoán y = w^T x + b | Linear Regression | `predict()` / `forward()` trong `linear_regression.py` |')
    lines.append('| Cập nhật trọng số bằng Gradient Descent | Linear Regression | `fit()` và `update_weights()` trong `linear_regression.py` |')
    lines.append('| Tạo mô hình baseline và so sánh với mô hình học được | Mean Baseline | `mean_baseline()` trong `metrics.py` |')
    lines.append('| Tách dữ liệu huấn luyện và kiểm thử | Preprocessing | `custom_train_test_split()` trong `preprocessing.py` |')
    lines.append('| Chuẩn hóa và mã hóa đặc trưng | Preprocessing | `fit_numeric_medians()`, `apply_numeric_medians()`, `onehot_encode_train_test()` trong `preprocessing.py` |')
    lines.append('| Đánh giá mô hình bằng MAE/RMSE/R2 | Metrics | `mae()`, `rmse()`, `r2_score()` trong `metrics.py` |')
    lines.append('| Trực quan hóa trực tiếp các điều cần kiểm tra | Visualization | `plot_training_loss()`, `plot_actual_vs_predicted()`, `plot_residuals()`, `plot_residual_distribution()` trong `visualization.py` |')
    lines.append('')
    lines.append('### 6.5. Tiền xử lý và ràng buộc sớm')
    lines.append('Dữ liệu được đọc trực tiếp từ `energy_efficiency_building_heating_cooling_load_dataset.csv`. Trong quá trình preprocessing, Y2 được loại bỏ khỏi khung đặc trưng X để tránh hiện tượng data leakage. Tập train/test được tách trước khi imputation, one-hot và scaling áp dụng trên tập train; sau đó, các phép biến đổi được áp dụng thống nhất lên tập test. Đây là một quy trình an toàn dạng train-only fit và test-only transform, và chỉ mục tiêu Y1 được tối ưu hóa bởi mô hình.')
    lines.append('')
    lines.append('### 6.8. Tiêu chí đánh giá chất lượng Prompting Logs')
    lines.append('Độ chuẩn xác kỹ thuật của prompt, khả năng kiểm soát kết quả, số lần phản hồi điều chỉnh, mức độ thấu hiểu code của người lập trình, đồng thời khả năng truy trace từ biểu thức toán học đến source code. Một prompt chất lượng tốt phải giúp con người kiểm tra mối nối giữa công thức học máy và mã nguồn được mô tả rõ ràng, có khả năng tái chạy được và chịu được thử nghiệm unit. Trong dự án này, chất lượng log được xác định qua việc nguồn dữ liệu rõ, mục tiêu Y1 rõ, Y2 bị loại ‘không cho vào X’, và mô hình custom NumPy được kiểm chứng bằng test.')
    lines.append('')
    lines.append('### 6.9. Cấu trúc Source Code thuật toán – Linear Regression')
    lines.append('Cấu trúc mã nguồn của thuật toán Linear Regression từ scratch theo đúng quy trình học máy được tổ chức thành các module chính: `data_loader.py` đọc dữ liệu; `eda.py` và `stage2_cleaning_eda.py` kiểm tra dữ liệu và EDA; `preprocessing.py` thực hiện train/test split, imputation, one-hot, và chuẩn hóa; `linear_regression.py` chứa lớp `LinearRegressionScratch` dùng Gradient Descent để cập nhật trọng số và bias; `metrics.py` chứa MAE, RMSE, R2 và baseline; `visualization.py` sinh các plot quan sát; `stage5_training_eval.py` thực hiện huấn luyện và đánh giá; `stage6_report.py` sinh báo cáo giải thích tham số; `stage7_final_artifacts.py` đóng gói văn bản và notebook cuối cùng.')
    lines.append('')
    lines.append('### 6.10. Bảng đối chiếu lý thuyết toán học và hàm mã nguồn tương ứng')
    lines.append('')
    lines.append('| Nội dung toán học cốt lõi | Thuật toán | Tên hàm / khối xử lý trong source code |')
    lines.append('|---|---|---|')
    lines.append('| Dự đoán y = w^T x + b | Linear Regression | `predict()` / `forward()` trong `linear_regression.py` |')
    lines.append('| Cập nhật trọng số bằng Gradient Descent | Gradient Descent | `fit()` trong `LinearRegressionScratch` và `loss_history` |')
    lines.append('| Tính ma trận hệ số và bias | Linear algebra | `get_coefficients()` và `get_intercept()` trong `linear_regression.py` |')
    lines.append('| Tách train/test trước xử lý | Data splitting | `custom_train_test_split()` trong `preprocessing.py` |')
    lines.append('| Chuyển đổi biến phân loại | One-hot encoding | `onehot_encode_train_test()` trong `preprocessing.py` |')
    lines.append('| Điền thiếu bằng trung vị train-only | Imputation | `fit_numeric_medians()` và `apply_numeric_medians()` trong `preprocessing.py` |')
    lines.append('| Chuẩn hóa đặc trưng bằng train mean/std | Scaling | `preprocess_pipeline()` trong `preprocessing.py` |')
    lines.append('| Đánh giá hiệu quả mô hình | MAE/RMSE/R2 | `mae()`, `rmse()`, `r2_score()` trong `metrics.py` |')
    lines.append('| So sánh với baseline mean | Mean Baseline | `mean_baseline()` trong `metrics.py` |')
    lines.append('| Curve học loss | Optimization trace | `plot_training_loss()` trong `visualization.py` |')
    lines.append('| Đồ thị thực tế và dự đoán | Residual diagnostics | `plot_actual_vs_predicted()` và `plot_residuals()` trong `visualization.py` |')
    lines.append('')
    lines.append('## Outputs')
    lines.append('- Figures saved under `outputs/figures/`')
    lines.append('- Metrics report saved under `outputs/tables/metrics_report.txt`')
    lines.append('- Coefficient table saved under `outputs/tables/model_coefficients.csv`')
    lines.append('- Narrative report saved under `outputs/stage6_report.txt`')
    lines.append('')

    final_report = '\n'.join(lines)
    return final_report


def generate_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Energy Efficiency Linear Regression\n",
                    "\n",
                    "This notebook reproduces the Stage 5 training/evaluation workflow and the Stage 6 interpretation artifacts.\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "PROJECT_ROOT = Path.cwd()\n",
                    "sys.path.insert(0, str(PROJECT_ROOT / 'src'))\n",
                    "from data_loader import load_dataset\n",
                    "from preprocessing import preprocess_pipeline\n",
                    "from linear_regression import LinearRegressionScratch\n",
                    "from metrics import mae, rmse, r2_score\n",
                    "\n",
                    "data_path = PROJECT_ROOT / 'data' / 'energy_efficiency_building_heating_cooling_load_dataset.csv'\n",
                    "df = load_dataset(data_path)\n",
                    "if 'Y2' in df.columns:\n",
                    "    df = df.drop(columns=['Y2'])\n",
                    "X_train, X_test, y_train, y_test, medians, scale_info = preprocess_pipeline(df, target_col='Y1', test_size=0.2, random_seed=42)\n",
                    "model = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000, tolerance=1e-6)\n",
                    "model.fit(X_train.values, y_train)\n",
                    "y_pred = model.predict(X_test.values)\n",
                    "baseline_pred = pd.Series(y_train).mean()\n",
                    "print('R2 scratch=', r2_score(y_test, y_pred))\n",
                    "print('MAE scratch=', mae(y_test, y_pred))\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "metrics = pd.read_csv(PROJECT_ROOT / 'outputs' / 'tables' / 'metrics_report.txt', sep=',', header=None)\n",
                    "metrics\n"
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


def main():
    # Ensure output directories exist.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = parse_metrics_report(TABLE_DIR / 'metrics_report.txt')
    coeff_df = parse_coefficients(TABLE_DIR / 'model_coefficients.csv')
    final_md = generate_final_markdown(metrics_df, coeff_df)

    # Write final markdown report.
    (OUT_DIR / 'final_report.md').write_text(final_md, encoding='utf-8')

    # Write notebook.
    notebook = generate_notebook()
    (NOTEBOOK_DIR / 'energy_efficiency_linear_regression_analysis.ipynb').write_text(
        json.dumps(notebook, indent=1),
        encoding='utf-8',
    )

    print('Stage 7 artifacts generated:')
    print(OUT_DIR / 'final_report.md')
    print(NOTEBOOK_DIR / 'energy_efficiency_linear_regression_analysis.ipynb')


if __name__ == '__main__':
    main()
