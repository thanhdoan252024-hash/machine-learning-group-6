import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'energy_efficiency_building_heating_cooling_load_dataset.csv'
OUT_DIR = PROJECT_ROOT / 'outputs'
FIG_DIR = OUT_DIR / 'figures'
TABLE_DIR = OUT_DIR / 'tables'

sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data_loader import load_dataset
from preprocessing import preprocess_pipeline
from linear_regression import LinearRegressionScratch
from metrics import rmse, mae, r2_score
from visualization import plot_training_loss, plot_actual_vs_predicted, plot_residuals, plot_residual_distribution


def evaluate_target(df, target_col, label):
    """Train and evaluate a Y1 heating-load or Y2 cooling-load model."""
    X_train, X_test, y_train, y_test, medians, scale_info = preprocess_pipeline(
        df,
        target_col=target_col,
        test_size=0.2,
        random_seed=42,
    )

    y_train_mean = np.mean(y_train)
    y_pred_base = np.full(y_test.shape, y_train_mean)

    lr = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000, tolerance=1e-6)
    lr.fit(X_train.values, y_train)
    y_pred_lr = lr.predict(X_test.values)

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

    # Write plot artifacts for this target.
    plot_training_loss(lr.loss_history, str(FIG_DIR / f'{label}_training_loss_curve.png'))
    plot_actual_vs_predicted(y_test, y_pred_lr, str(FIG_DIR / f'{label}_actual_vs_predicted.png'))
    plot_residuals(y_pred_lr, y_test, str(FIG_DIR / f'{label}_residual_plot.png'))
    plot_residual_distribution(y_test - y_pred_lr, str(FIG_DIR / f'{label}_residual_distribution.png'))

    return metrics, lr, X_train, X_test, y_train, y_test


def main():
    df = load_dataset(DATA_PATH)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = []
    with open(TABLE_DIR / 'metrics_report.txt', 'w', encoding='utf-8') as f:
        f.write('Model,MAE,RMSE,R2\n')

    results = {}
    for target_col, label, label_full in [
        ('Y1', 'heating', 'Heating Load (Y1)'),
        ('Y2', 'cooling', 'Cooling Load (Y2)'),
    ]:
        target_df = df.copy()
        if target_col == 'Y1' and 'Y2' in target_df.columns:
            target_df = target_df.drop(columns=['Y2'])
        elif target_col == 'Y2' and 'Y1' in target_df.columns:
            target_df = target_df.drop(columns=['Y1'])

        metrics, lr, X_train, X_test, y_train, y_test = evaluate_target(target_df, target_col, label)
        results[target_col] = {
            'metrics': metrics,
            'lr': lr,
            'X_train': X_train,
            'y_train': y_train,
            'y_test': y_test,
        }

        with open(TABLE_DIR / 'metrics_report.txt', 'a', encoding='utf-8') as f:
            f.write(f'{label_full} Mean Baseline,{metrics["Mean Baseline"]["MAE"]},{metrics["Mean Baseline"]["RMSE"]},{metrics["Mean Baseline"]["R2"]}\n')
            f.write(f'{label_full} Linear Regression,{metrics["Linear Regression"]["MAE"]},{metrics["Linear Regression"]["RMSE"]},{metrics["Linear Regression"]["R2"]}\n')

    with open(OUT_DIR / 'stage5_report.txt', 'w', encoding='utf-8') as f:
        f.write('=== Stage 5: Linear Regression Training & Evaluation ===\n')
        f.write(f'Rows loaded: {df.shape[0]}\n')
        f.write('Targets trained: Y1 Heating Load and Y2 Cooling Load.\n')
        f.write('Leakage guard: the opposite target is excluded from X for each run.\n')
        f.write('No sklearn LinearRegression used.\n\n')

        for target_col in ['Y1', 'Y2']:
            target_label = 'Heating Load (Y1)' if target_col == 'Y1' else 'Cooling Load (Y2)'
            metrics = results[target_col]['metrics']
            lr = results[target_col]['lr']
            X_train = results[target_col]['X_train']
            y_train = results[target_col]['y_train']
            y_test = results[target_col]['y_test']
            f.write(f'=== Target {target_col}: {target_label} ===\n')
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
            f.write('\n')

    print('=== Stage 5 complete ===')
    for target_col in ['Y1', 'Y2']:
        metrics = results[target_col]['metrics']
        print(f'{target_col} Baseline MAE={metrics["Mean Baseline"]["MAE"]}; RMSE={metrics["Mean Baseline"]["RMSE"]}; R2={metrics["Mean Baseline"]["R2"]}')
        print(f'{target_col} LR MAE={metrics["Linear Regression"]["MAE"]}; RMSE={metrics["Linear Regression"]["RMSE"]}; R2={metrics["Linear Regression"]["R2"]}')
    print('Figures saved:', FIG_DIR)
    print('Metrics report saved:', TABLE_DIR / 'metrics_report.txt')


if __name__ == '__main__':
    main()
