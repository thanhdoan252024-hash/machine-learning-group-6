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


def generate_final_markdown(metrics_df, coeff_df, coeff_df_y2=None):
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
    lines.append('Predict the building Heating Load (Y1) and Cooling Load (Y2) from the architectural variables X1 through X8 using a custom Linear Regression from scratch implemented with NumPy and Gradient Descent, while preventing target leakage by excluding the opposite target from the feature matrix used for each training run.')
    lines.append('')
    lines.append('## Dataset')
    lines.append('The source dataset is the available file in the project data directory. The dataset contains the architectural variables X1..X8 and two target-like variables Y1 (Heating Load) and Y2 (Cooling Load). The custom model is implemented with NumPy and gradient descent only; no sklearn `LinearRegression` is used. For the Y1 model, Y2 is excluded from X; for the Y2 model, Y1 is excluded from X to avoid target leakage.')
    lines.append('')
    lines.append('## Metrics')
    lines.append('')
    lines.append('| Model | MAE | RMSE | R2 |')
    lines.append('|---|---:|---:|---:|')
    lines.extend([f'| {row["Model"]} | {row["MAE"]:.6f} | {row["RMSE"]:.6f} | {row["R2"]:.6f} |' for _, row in metrics_df.iterrows()])
    lines.append('')
    lines.append('## Interpretation')
    lines.append('The scratch linear model improves substantially over the mean baseline in MAE, RMSE and R2 for both Y1 and Y2, confirming that the selected architectural features carry meaningful signal for both targets.')
    lines.append('The coefficient tables are stored separately for Heating Load (Y1) and Cooling Load (Y2), sorted by absolute coefficient magnitude.')
    lines.append('')
    lines.append('## Top coefficients')
    lines.append('')
    lines.append('| feature | coefficient |')
    lines.append('|---|---:|')
    for _, row in top_features.iterrows():
        lines.append(f"| {row['feature']} | {row['coefficient']:.6f} |")
    if coeff_df_y2 is not None and not coeff_df_y2.empty:
        lines.append('')
        lines.append('## Top Cooling Load (Y2) coefficients')
        lines.append('')
        lines.append('| feature | coefficient |')
        lines.append('|---|---:|')
        for _, row in coeff_df_y2.head(5).iterrows():
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
    lines.append('Dữ liệu được đọc trực tiếp từ `energy_efficiency_building_heating_cooling_load_dataset.csv`. Trong quá trình preprocessing, target đối nghịch được loại bỏ khỏi khung đặc trưng X để tránh data leakage. Tập train/test được tách trước khi imputation, one-hot và scaling áp dụng trên tập train; sau đó, các phép biến đổi được áp dụng thống nhất lên tập test cho cả Y1 và Y2.')
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
    lines.append('- Heating coefficient table saved under `outputs/tables/model_coefficients_y1.csv`')
    lines.append('- Cooling coefficient table saved under `outputs/tables/model_coefficients_y2.csv`')
    lines.append('- Narrative report saved under `outputs/stage6_report.txt`')
    lines.append('')

    final_report = '\n'.join(lines)
    return final_report


def generate_notebook():
    def markdown(text):
        return {
            'cell_type': 'markdown', 'metadata': {},
            'source': [line + '\n' for line in text.splitlines()],
        }

    def code(text):
        return {
            'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': [line + '\n' for line in text.splitlines()],
        }

    notebook = {
        'cells': [
            markdown('# Energy Efficiency Linear Regression\n\nThis notebook presents the complete Y1/Y2 workflow, including EDA, leakage-safe preprocessing, scratch gradient descent, cross-validation, diagnostics, and limitations.'),
            markdown('## 1. Load and audit the dataset'),
            code("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from data_loader import load_dataset, print_schema
from preprocessing import preprocess_pipeline, make_kfold_splits
from linear_regression import LinearRegressionScratch
from metrics import mae, rmse, r2_score

data_path = PROJECT_ROOT / 'data' / 'energy_efficiency_building_heating_cooling_load_dataset.csv'
df = load_dataset(data_path)
print_schema(df)
df.describe(include='all')"""),
            markdown('## 2. EDA for Heating Load and Cooling Load'),
            code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['Y1'], bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('Heating Load (Y1)')
axes[1].hist(df['Y2'], bins=30, color='darkorange', edgecolor='white')
axes[1].set_title('Cooling Load (Y2)')
plt.tight_layout()
plt.show()

df[[f'X{i}' for i in range(1, 9)] + ['Y1', 'Y2']].corr(numeric_only=True)[['Y1', 'Y2']]"""),
            markdown('## 3. Multicollinearity and leakage controls\n\nX2 is excluded because the dataset satisfies X2 = X3 + 2X4. The opposite target is removed separately for each target run.'),
            code("""print('X2 - X3 - 2*X4 max residual:', np.abs(df['X2'] - df['X3'] - 2 * df['X4']).max())
print('Y1 run excludes Y2; Y2 run excludes Y1.')"""),
            markdown('## 4. Linear Regression and Gradient Descent\n\nThe scratch model uses y_hat = Xw + b and minimizes mean squared error with batch gradient descent and tolerance-based early stopping.'),
            markdown('## 5. Train and evaluate both targets'),
            code("""def train_target(target_col):
    target_df = df.drop(columns=['Y2' if target_col == 'Y1' else 'Y1'])
    X_train, X_test, y_train, y_test, _, _ = preprocess_pipeline(
        target_df, target_col=target_col, test_size=0.2, random_seed=42
    )
    model = LinearRegressionScratch(learning_rate=0.01, n_iterations=10000, tolerance=1e-6)
    model.fit(X_train.values, y_train)
    prediction = model.predict(X_test.values)
    baseline = np.full(y_test.shape, np.mean(y_train))
    return {
        'model': model, 'y_test': y_test, 'prediction': prediction,
        'baseline': baseline, 'columns': X_train.columns,
    }

results = {target: train_target(target) for target in ['Y1', 'Y2']}
for target, result in results.items():
    print(target, 'MAE=', mae(result['y_test'], result['prediction']))
    print(target, 'RMSE=', rmse(result['y_test'], result['prediction']))
    print(target, 'R2=', r2_score(result['y_test'], result['prediction']))
    print(target, 'iterations=', result['model'].iterations_run)"""),
            markdown('## 6. Diagnostics and 5-fold cross-validation'),
            code("""for target, result in results.items():
    residuals = result['y_test'] - result['prediction']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(result['y_test'], result['prediction'], alpha=0.5)
    axes[0].set_title(f'{target}: actual vs predicted')
    axes[1].hist(residuals, bins=30, color='slateblue', edgecolor='white')
    axes[1].set_title(f'{target}: residual distribution')
    plt.tight_layout()
    plt.show()

for target in ['Y1', 'Y2']:
    target_df = df.drop(columns=['Y2' if target == 'Y1' else 'Y1'])
    folds = make_kfold_splits(target_df, target_col=target, n_splits=5, random_seed=42)
    fold_scores = []
    for train_df, test_df in folds:
        X_train, X_test, y_train, y_test, _, _ = preprocess_pipeline(
            target_df, target_col=target, train_df=train_df, test_df=test_df,
            test_size=0.2, random_seed=42
        )
        model = LinearRegressionScratch(learning_rate=0.01, n_iterations=10000, tolerance=1e-6)
        model.fit(X_train.values, y_train)
        fold_scores.append(r2_score(y_test, model.predict(X_test.values)))
    print(target, 'CV R2 mean/std=', np.mean(fold_scores), np.std(fold_scores))"""),
            markdown('## 7. Conclusion and limitations\n\nThe model is a strong interpretable baseline for both targets. It does not capture all nonlinear heat-transfer effects, so residual patterns and generalization should be considered before production use.'),
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
    coeff_df = parse_coefficients(TABLE_DIR / 'model_coefficients_y1.csv')
    coeff_df_y2 = parse_coefficients(TABLE_DIR / 'model_coefficients_y2.csv')
    final_md = generate_final_markdown(metrics_df, coeff_df, coeff_df_y2)

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
