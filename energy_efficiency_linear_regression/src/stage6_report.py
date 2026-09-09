import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'energy_efficiency_building_heating_cooling_load_dataset.csv'
OUT_DIR = PROJECT_ROOT / 'outputs'
FIG_DIR = OUT_DIR / 'figures'
TABLE_DIR = OUT_DIR / 'tables'

sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data_loader import load_dataset
from preprocessing import preprocess_pipeline
from linear_regression import LinearRegressionScratch
from metrics import mae, rmse, r2_score


def build_stage6_report():
    """
    Stage 6 report and interpretation artifact generator.
    Produces:
      - coefficients table (csv)
      - summary interpretive report (txt)
    """
    df = load_dataset(DATA_PATH)

    # Leakage guard: Y2 must not appear in the final X matrix.
    if 'Y2' in df.columns:
        df = df.drop(columns=['Y2'])

    X_train, X_test, y_train, y_test, medians, scale_info = preprocess_pipeline(
        df,
        target_col='Y1',
        test_size=0.2,
        random_seed=42,
    )

    model = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000, tolerance=1e-6)
    model.fit(X_train.values, y_train)

    y_pred = model.predict(X_test.values)

    baseline_y_pred = np.full_like(y_test, float(np.mean(y_train)), dtype=float)

    metrics = {
        'Baseline Mean': {
            'MAE': mae(y_test, baseline_y_pred),
            'RMSE': rmse(y_test, baseline_y_pred),
            'R2': r2_score(y_test, baseline_y_pred),
        },
        'Linear Regression Scratch': {
            'MAE': mae(y_test, y_pred),
            'RMSE': rmse(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
        },
    }

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    coeff_table = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': model.get_coefficients(),
        'absolute_coefficient': np.abs(model.get_coefficients()),
    })
    coeff_table = coeff_table.sort_values('absolute_coefficient', ascending=False)
    coeff_table.to_csv(TABLE_DIR / 'model_coefficients.csv', index=False)

    # Save a short, human-readable interpretation report.
    top_features = coeff_table.head(5)
    top_positive = coeff_table.sort_values('coefficient', ascending=False).head(3)
    top_negative = coeff_table.sort_values('coefficient', ascending=True).head(3)

    with open(OUT_DIR / 'stage6_report.txt', 'w', encoding='utf-8') as f:
        f.write('=== Stage 6: Linear Regression Report and Interpretation ===\n')
        f.write('Project: Energy Efficiency Building Performance Prediction\n')
        f.write('Target: Y1 (Heating Load)\n')
        f.write('Leakage guard: Y2 is excluded from design matrix X.\n')
        f.write('Training approach: custom NumPy scratch LinearRegressionScratch with Gradient Descent.\n')
        f.write('\nData shape: {} rows, {} columns after Y2 removal.\n'.format(df.shape[0], df.shape[1]))
        f.write('Train rows: {}\n'.format(len(y_train)))
        f.write('Test rows: {}\n'.format(len(y_test)))
        f.write('Model learning_rate: {}\n'.format(model.learning_rate))
        f.write('Iterations run: {}\n'.format(model.iterations_run))
        f.write('Final training loss: {:.6f}\n'.format(model.loss_history[-1]))
        f.write('\nMetric comparison\n')
        f.write('Baseline Mean -- MAE: {:.6f}, RMSE: {:.6f}, R2: {:.6f}\n'.format(
            metrics['Baseline Mean']['MAE'], metrics['Baseline Mean']['RMSE'], metrics['Baseline Mean']['R2']
        ))
        f.write('Linear Regression Scratch -- MAE: {:.6f}, RMSE: {:.6f}, R2: {:.6f}\n'.format(
            metrics['Linear Regression Scratch']['MAE'], metrics['Linear Regression Scratch']['RMSE'], metrics['Linear Regression Scratch']['R2']
        ))

        f.write('\nCoefficient interpretation\n')
        f.write('Feature coefficients are standardized by the preprocessing pipeline.\n')
        f.write('Top 5 features by absolute coefficient magnitude:\n')
        for _, row in top_features.iterrows():
            f.write('- {}: coefficient = {:.6f}\n'.format(row['feature'], row['coefficient']))

        f.write('\nLargest positive coefficients:\n')
        for _, row in top_positive.iterrows():
            f.write('- {}: {:.6f}\n'.format(row['feature'], row['coefficient']))

        f.write('\nLargest negative coefficients:\n')
        for _, row in top_negative.iterrows():
            f.write('- {}: {:.6f}\n'.format(row['feature'], row['coefficient']))

        f.write('\nBusiness/technical interpretation:\n')
        f.write('The model provides an interpretable linear approximation of heating load using the architectural predictors available in the dataset.\n')
        f.write('Feature coefficients reveal relative directional strength. Positive coefficients indicate higher predicted Y1 as the standardized feature increases, while negative coefficients indicate the reverse relationship.\n')
        f.write('The model is suitable as a baseline analytical model, though nonlinear physical building heat transfer effects may remain.\n')

    print('Stage 6 artifacts generated:')
    print(TABLE_DIR / 'model_coefficients.csv')
    print(OUT_DIR / 'stage6_report.txt')


if __name__ == '__main__':
    build_stage6_report()
