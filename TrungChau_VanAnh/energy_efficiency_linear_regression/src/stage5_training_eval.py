import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'energy_efficiency_building_heating_cooling_load_dataset.csv'
OUT_DIR = PROJECT_ROOT / 'outputs'
FIG_DIR = OUT_DIR / 'figures'
TABLE_DIR = OUT_DIR / 'tables'

sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data_loader import load_dataset
from preprocessing import preprocess_pipeline
from linear_regression import LinearRegressionScratch
from metrics import mse, rmse, mae, r2_score, mean_baseline
from visualization import plot_training_loss, plot_actual_vs_predicted, plot_residuals, plot_residual_distribution


def main():
    # Load dataset
    df = load_dataset(DATA_PATH)

    # Leakage-safe feature matrix excludes Y2
    if 'Y2' in df.columns:
        df = df.drop(columns=['Y2'])

    # Split & preprocess in the required order
    X_train, X_test, y_train, y_test, medians, scale_info = preprocess_pipeline(
        df,
        target_col='Y1',
        test_size=0.2,
        random_seed=42,
    )

    # Baseline mean predictor
    y_train_mean = np.mean(y_train)
    y_pred_base = np.full(y_test.shape, y_train_mean)

    # Build custom linear regression from scratch
    lr = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000, tolerance=1e-6)
    lr.fit(X_train.values, y_train)

    y_pred_lr = lr.predict(X_test.values)

    # Metrics
    metrics = {
        'Mean Baseline': {
            'MAE': mae(y_test, y_pred_base),
            'RMSE': rmse(y_test, y_pred_base),
            'R2': r2_score(y_test, y_pred_base),
        },
        'Linear Regression': {
            'MAE': mae(y_test, y_pred_lr),
            'RMSE': rmse(y_test, y_pred_lr),
            'R2': r2_score(y_test, y_pred_lr),
        },
    }

    # Save metric table
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLE_DIR / 'metrics_report.txt', 'w', encoding='utf-8') as f:
        f.write('Model,MAE,RMSE,R2\n')
        for model, vals in metrics.items():
            f.write(f'{model},{vals["MAE"]},{vals["RMSE"]},{vals["R2"]}\n')

    # Save summary report
    with open(OUT_DIR / 'stage5_report.txt', 'w', encoding='utf-8') as f:
        f.write('=== Stage 5: Linear Regression Training & Evaluation ===\n')
        f.write(f'Rows loaded: {df.shape[0]}\n')
        f.write(f'Final feature matrix columns: {list(X_train.columns)}\n')
        f.write(f'Train rows: {len(y_train)}\n')
        f.write(f'Test rows: {len(y_test)}\n')
        f.write(f'Learning rate: {lr.learning_rate}\n')
        f.write(f'Iterations run: {lr.iterations_run}\n')
        f.write(f'Final loss: {lr.loss_history[-1]}\n')
        f.write(f'Baseline MAE: {metrics["Mean Baseline"]["MAE"]}\n')
        f.write(f'Baseline RMSE: {metrics["Mean Baseline"]["RMSE"]}\n')
        f.write(f'Baseline R2: {metrics["Mean Baseline"]["R2"]}\n')
        f.write(f'LR MAE: {metrics["Linear Regression"]["MAE"]}\n')
        f.write(f'LR RMSE: {metrics["Linear Regression"]["RMSE"]}\n')
        f.write(f'LR R2: {metrics["Linear Regression"]["R2"]}\n')
        f.write('Leakage guard: Y2 excluded from X.\n')
        f.write('No sklearn LinearRegression used.\n')

    # Generate required plots
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_loss(lr.loss_history, str(FIG_DIR / 'training_loss_curve.png'))
    plot_actual_vs_predicted(y_test, y_pred_lr, str(FIG_DIR / 'actual_vs_predicted.png'))
    plot_residuals(y_pred_lr, y_test, str(FIG_DIR / 'residual_plot.png'))
    plot_residual_distribution(y_test - y_pred_lr, str(FIG_DIR / 'residual_distribution.png'))

    print('=== Stage 5 complete ===')
    print(f'Baseline MAE={metrics["Mean Baseline"]["MAE"]}; RMSE={metrics["Mean Baseline"]["RMSE"]}; R2={metrics["Mean Baseline"]["R2"]}')
    print(f'LR MAE={metrics["Linear Regression"]["MAE"]}; RMSE={metrics["Linear Regression"]["RMSE"]}; R2={metrics["Linear Regression"]["R2"]}')
    print('Figures saved:', FIG_DIR)
    print('Metrics report saved:', TABLE_DIR / 'metrics_report.txt')


if __name__ == '__main__':
    main()
