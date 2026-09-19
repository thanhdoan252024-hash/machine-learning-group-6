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
from preprocessing import preprocess_pipeline, make_kfold_splits
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

    lr = LinearRegressionScratch(learning_rate=0.01, n_iterations=10000, tolerance=1e-6)
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

    # 5-fold cross-validation on the same target using leakage-safe folds.
    folds = make_kfold_splits(df, target_col=target_col, n_splits=5, random_seed=42)
    cv_mae, cv_rmse, cv_r2 = [], [], []
    for idx, (train_df, test_df) in enumerate(folds, start=1):
        X_train_cv, X_test_cv, y_train_cv, y_test_cv, _, _ = preprocess_pipeline(
            df=df,
            target_col=target_col,
            train_df=train_df,
            test_df=test_df,
            test_size=0.2,
            random_seed=42,
        )
        model_cv = LinearRegressionScratch(learning_rate=0.01, n_iterations=10000, tolerance=1e-6)
        model_cv.fit(X_train_cv.values, y_train_cv)
        y_pred_cv = model_cv.predict(X_test_cv.values)
        cv_mae.append(mae(y_test_cv, y_pred_cv))
        cv_rmse.append(rmse(y_test_cv, y_pred_cv))
        cv_r2.append(r2_score(y_test_cv, y_pred_cv))

    cv = {
        'MAE_mean': float(np.mean(cv_mae)),
        'MAE_std': float(np.std(cv_mae)),
        'RMSE_mean': float(np.mean(cv_rmse)),
        'RMSE_std': float(np.std(cv_rmse)),
        'R2_mean': float(np.mean(cv_r2)),
        'R2_std': float(np.std(cv_r2)),
    }

    # Write plot artifacts for this target.
    target_name = 'Heating Load (Y1)' if target_col == 'Y1' else 'Cooling Load (Y2)'
    plot_training_loss(lr.loss_history, str(FIG_DIR / f'{label}_training_loss_curve.png'), target_name)
    plot_actual_vs_predicted(y_test, y_pred_lr, str(FIG_DIR / f'{label}_actual_vs_predicted.png'), target_name)
    plot_residuals(y_pred_lr, y_test, str(FIG_DIR / f'{label}_residual_plot.png'), target_name)
    plot_residual_distribution(y_test - y_pred_lr, str(FIG_DIR / f'{label}_residual_distribution.png'), target_name)

    return metrics, lr, X_train, X_test, y_train, y_test, cv


def main():
    df = load_dataset(DATA_PATH)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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

        metrics, lr, X_train, X_test, y_train, y_test, cv = evaluate_target(target_df, target_col, label)
        results[target_col] = {
            'metrics': metrics,
            'lr': lr,
            'X_train': X_train,
            'y_train': y_train,
            'y_test': y_test,
            'cv': cv,
        }

        with open(TABLE_DIR / 'metrics_report.txt', 'a', encoding='utf-8') as f:
            f.write(f'{label_full} Mean Baseline,{metrics["Mean Baseline"]["MAE"]},{metrics["Mean Baseline"]["RMSE"]},{metrics["Mean Baseline"]["R2"]}\n')
            f.write(f'{label_full} Linear Regression,{metrics["Linear Regression"]["MAE"]},{metrics["Linear Regression"]["RMSE"]},{metrics["Linear Regression"]["R2"]}\n')

    with open(OUT_DIR / 'stage5_report.txt', 'w', encoding='utf-8') as f:
        f.write('=== Stage 5: Linear Regression Training & Evaluation ===\n')
        f.write(f'Rows loaded: {df.shape[0]}\n')
        f.write('Targets trained: Y1 Heating Load and Y2 Cooling Load.\n')
        f.write('Leakage guard: the opposite target is excluded from X for each run.\n')
        f.write('Multicollinearity guard: X2 has been removed because X2 = X3 + 2X4 in the data.\n')
        f.write('Cross-validation guard: 5-fold CV is reported for each target with mean/std.\n')
        f.write('Convergence guard: maximum 10000 iterations with tolerance-based early stopping.\n')
        f.write('No sklearn LinearRegression used.\n\n')

        for target_col in ['Y1', 'Y2']:
            target_label = 'Heating Load (Y1)' if target_col == 'Y1' else 'Cooling Load (Y2)'
            metrics = results[target_col]['metrics']
            lr = results[target_col]['lr']
            X_train = results[target_col]['X_train']
            y_train = results[target_col]['y_train']
            y_test = results[target_col]['y_test']
            cv = results[target_col]['cv']
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
            f.write(f'CV MAE_mean: {cv["MAE_mean"]:.6f}, MAE_std: {cv["MAE_std"]:.6f}\n')
            f.write(f'CV RMSE_mean: {cv["RMSE_mean"]:.6f}, RMSE_std: {cv["RMSE_std"]:.6f}\n')
            f.write(f'CV R2_mean: {cv["R2_mean"]:.6f}, R2_std: {cv["R2_std"]:.6f}\n')
            f.write('\n')

    print('=== Stage 5 complete ===')
    for target_col in ['Y1', 'Y2']:
        metrics = results[target_col]['metrics']
        cv = results[target_col]['cv']
        print(f'{target_col} Baseline MAE={metrics["Mean Baseline"]["MAE"]}; RMSE={metrics["Mean Baseline"]["RMSE"]}; R2={metrics["Mean Baseline"]["R2"]}')
        print(f'{target_col} LR MAE={metrics["Linear Regression"]["MAE"]}; RMSE={metrics["Linear Regression"]["RMSE"]}; R2={metrics["Linear Regression"]["R2"]}')
        print(f'{target_col} CV MAE mean/std = {cv["MAE_mean"]:.6f}/{cv["MAE_std"]:.6f}; RMSE mean/std = {cv["RMSE_mean"]:.6f}/{cv["RMSE_std"]:.6f}; R2 mean/std = {cv["R2_mean"]:.6f}/{cv["R2_std"]:.6f}')
    print('Figures saved:', FIG_DIR)
    print('Metrics report saved:', TABLE_DIR / 'metrics_report.txt')


if __name__ == '__main__':
    main()
