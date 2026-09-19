import sys
from pathlib import Path
import subprocess
import csv

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
    return path.exists() and path.stat().st_size > 0


def read_metrics_report(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))

    required_models = {
        'Heating Load (Y1) Mean Baseline',
        'Heating Load (Y1) Linear Regression',
        'Cooling Load (Y2) Mean Baseline',
        'Cooling Load (Y2) Linear Regression',
    }
    actual_models = {row.get('Model') for row in rows}
    if actual_models != required_models:
        raise AssertionError(
            'metrics_report.txt must contain exactly the four Y1/Y2 baseline and '
            f'linear-regression rows; found {sorted(actual_models)}'
        )

    for row in rows:
        for metric in ('MAE', 'RMSE', 'R2'):
            try:
                float(row[metric])
            except (KeyError, TypeError, ValueError) as exc:
                raise AssertionError(f'Invalid {metric} in metrics report: {row}') from exc

    return rows


def main():
    # 1. Basic artifact existence
    required_files = [
        DATA_PATH,
        TABLE_DIR / 'metrics_report.txt',
        TABLE_DIR / 'model_coefficients.csv',
        TABLE_DIR / 'model_coefficients_y1.csv',
        TABLE_DIR / 'model_coefficients_y2.csv',
        OUT_DIR / 'stage2_report.txt',
        OUT_DIR / 'stage3_report.txt',
        OUT_DIR / 'stage5_report.txt',
        OUT_DIR / 'stage6_report.txt',
        OUT_DIR / 'final_report.md',
        NOTEBOOK_DIR / 'energy_efficiency_linear_regression_analysis.ipynb',
    ]

    required_figures = [
        f'{label}_{plot_type}.png'
        for label in ('heating', 'cooling')
        for plot_type in (
            'training_loss_curve',
            'actual_vs_predicted',
            'residual_plot',
            'residual_distribution',
        )
    ]

    # 2. Dataset audit and leakage rule check
    df = load_dataset(DATA_PATH)
    # 3. Validate preprocessing and model contracts for both targets.
    validation_metrics = {}
    for target_col, opposite_target in (('Y1', 'Y2'), ('Y2', 'Y1')):
        X_train, X_test, y_train, y_test, _, _ = preprocess_pipeline(
            df=df,
            target_col=target_col,
            test_size=0.2,
            random_seed=42,
        )
        if opposite_target in X_train.columns or opposite_target in X_test.columns:
            raise AssertionError(f'{opposite_target} leaked into {target_col} features.')

        model = LinearRegressionScratch(
            learning_rate=0.01,
            n_iterations=10000,
            tolerance=1e-6,
        )
        model.fit(X_train.values, y_train)
        y_pred = model.predict(X_test.values)
        validation_metrics[target_col] = {
            'MAE': mae(y_test, y_pred),
            'RMSE': rmse(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
        }

    failures = []
    try:
        metrics_rows = read_metrics_report(TABLE_DIR / 'metrics_report.txt')
    except AssertionError as exc:
        metrics_rows = []
        failures.append(str(exc))
    else:
        expected_lr = {
            'Heating Load (Y1) Linear Regression': validation_metrics['Y1'],
            'Cooling Load (Y2) Linear Regression': validation_metrics['Y2'],
        }
        for row in metrics_rows:
            if row['Model'] not in expected_lr:
                continue
            for metric in ('MAE', 'RMSE', 'R2'):
                if abs(float(row[metric]) - expected_lr[row['Model']][metric]) > 1e-6:
                    failures.append(
                        f'Metric mismatch for {row["Model"]} {metric}: '
                        f'report={row[metric]}, recomputed={expected_lr[row["Model"]][metric]}'
                    )

    # 4. Validate all fresh artifact names and contents.
    missing = []
    for f in required_files:
        if not assert_file(f):
            missing.append(str(f))

    for fig in required_figures:
        fig_path = FIG_DIR / fig
        if not assert_file(fig_path):
            missing.append(str(fig_path))
    failures.extend(f'Missing or empty artifact: {item}' for item in missing)

    # 5. Run tests and require a successful exit code.
    try:
        test_result = subprocess.run(
            [sys.executable, '-m', 'pytest', '-q'],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        test_result = None
        failures.append(f'Pytest could not be started: {exc}')
    else:
        if test_result.returncode != 0:
            failures.append(f'Pytest failed with exit_code={test_result.returncode}')

    # 6. Write validation report before failing, so failures are inspectable.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / 'stage8_validation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=== Stage 8: Validation & Final Quality Check ===\n')
        f.write('Dataset rows: {}\n'.format(df.shape[0]))
        f.write('Dataset columns: {}\n'.format(', '.join(df.columns)))
        for target_col, metrics in validation_metrics.items():
            f.write(
                f'{target_col} validation metrics: '
                f'MAE={metrics["MAE"]:.6f}, '
                f'RMSE={metrics["RMSE"]:.6f}, '
                f'R2={metrics["R2"]:.6f}\n'
            )
        f.write('Leakage check: Y1/Y2 opposite targets excluded. {}\n'.format(
            'PASS' if not any('leaked' in item for item in failures) else 'FAIL'
        ))
        f.write('Metrics report check: {}\n'.format('PASS' if metrics_rows else 'FAIL'))
        f.write('Artifact check: {}\n'.format('PASS' if not missing else 'FAIL'))
        if test_result is not None:
            f.write('Pytest check: exit_code={}\n'.format(test_result.returncode))
            if test_result.stdout:
                f.write(test_result.stdout)
            if test_result.stderr:
                f.write(test_result.stderr)
        else:
            f.write('Pytest check: skipped because pytest execution could not be started.\n')
        if failures:
            f.write('Overall validation: FAIL\n')
            for failure in failures:
                f.write(f'Failure: {failure}\n')
        else:
            f.write('Overall validation: PASS\n')

    if failures:
        raise AssertionError('Stage 8 validation failed: ' + '; '.join(failures))

    print('Stage 8 validation complete.')
    print('Validation report saved:', report_path)
    print('Validation metrics:', validation_metrics)


if __name__ == '__main__':
    main()
