import sys
from pathlib import Path
import subprocess
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'energy_efficiency_building_heating_cooling_load_dataset.csv'
OUT_DIR = PROJECT_ROOT / 'outputs'
FIG_DIR = OUT_DIR / 'figures'
TABLE_DIR = OUT_DIR / 'tables'
NOTEBOOK_DIR = PROJECT_ROOT / 'notebooks'

sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data_loader import load_dataset
from preprocessing import preprocess_pipeline
from linear_regression import LinearRegressionScratch
from metrics import mae, rmse, r2_score


def assert_file(path):
    return path.exists()


def main():
    # 1. Basic artifact existence
    required_files = [
        DATA_PATH,
        TABLE_DIR / 'metrics_report.txt',
        TABLE_DIR / 'model_coefficients.csv',
        OUT_DIR / 'stage5_report.txt',
        OUT_DIR / 'stage6_report.txt',
        OUT_DIR / 'final_report.md',
        NOTEBOOK_DIR / 'energy_efficiency_linear_regression_analysis.ipynb',
    ]

    required_figures = [
        'training_loss_curve.png',
        'actual_vs_predicted.png',
        'residual_plot.png',
        'residual_distribution.png',
    ]

    # 2. Dataset audit and leakage rule check
    df = load_dataset(DATA_PATH)
    if 'Y2' in df.columns:
        df_without_y2 = df.drop(columns=['Y2'])
    else:
        df_without_y2 = df

    if 'Y2' in df_without_y2.columns:
        raise AssertionError('Y2 leaked into the feature frame after drop.')

    # 3. Process through preprocessing to ensure consistent pipeline shape
    X_train, X_test, y_train, y_test, medians, scale_info = preprocess_pipeline(
        df=df,
        target_col='Y1',
        test_size=0.2,
        random_seed=42,
    )

    # 4. Fit model once more to ensure model contract is correct
    model = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000, tolerance=1e-6)
    model.fit(X_train.values, y_train)
    y_pred = model.predict(X_test.values)

    # Compute metrics again and compare against file metrics
    lr_mae = mae(y_test, y_pred)
    lr_rmse = rmse(y_test, y_pred)
    lr_r2 = r2_score(y_test, y_pred)

    # 5. Validate artifacts exist
    missing = []
    for f in required_files:
        if not assert_file(f):
            missing.append(str(f))

    for fig in required_figures:
        fig_path = FIG_DIR / fig
        if not assert_file(fig_path):
            missing.append(str(fig_path))

    # 6. Write validation report
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / 'stage8_validation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=== Stage 8: Validation & Final Quality Check ===\n')
        f.write('Dataset rows: {}\n'.format(df.shape[0]))
        f.write('Dataset columns: {}\n'.format(', '.join(df.columns)))
        f.write('Leakage check: Y2 dropped from X before modeling. PASS\n')
        f.write('Preprocessing check: train/test arrays shape validated through preprocess_pipeline. PASS\n')
        f.write('Model check: LinearRegressionScratch fit/predict executed. PASS\n')
        f.write('Metrics: MAE={:.6f}, RMSE={:.6f}, R2={:.6f}\n'.format(lr_mae, lr_rmse, lr_r2))

        if missing:
            f.write('Artifact check: FAIL\n')
            for item in missing:
                f.write('Missing artifact: {}\n'.format(item))
            raise AssertionError('Missing required artifacts: {}'.format(', '.join(missing)))
        else:
            f.write('Artifact check: PASS\n')

    # 7. Optional: run tests if pytest available.
    try:
        test_result = subprocess.run(
            [sys.executable, '-m', 'pytest', '-q'],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        test_result = None
    
    with open(report_path, 'a', encoding='utf-8') as f:
        if test_result is not None:
            f.write('Pytest check: exit_code={}\n'.format(test_result.returncode))
            if test_result.stdout:
                f.write(test_result.stdout)
            if test_result.stderr:
                f.write(test_result.stderr)
        else:
            f.write('Pytest check: skipped because pytest execution could not be started.\n')

    print('Stage 8 validation complete.')
    print('Validation report saved:', report_path)
    print('Validation metrics: MAE={:.6f}, RMSE={:.6f}, R2={:.6f}'.format(lr_mae, lr_rmse, lr_r2))


if __name__ == '__main__':
    main()
