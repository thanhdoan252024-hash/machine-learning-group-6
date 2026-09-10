import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'energy_efficiency_building_heating_cooling_load_dataset.csv'
OUT_DIR = PROJECT_ROOT / 'outputs'
FIG_DIR = OUT_DIR / 'figures'
TABLE_DIR = OUT_DIR / 'tables'

sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data_loader import load_dataset, print_schema
from data_cleaning import audit_target_y1, validate_feature_columns, check_physical_feature_ranges
from eda import basic_eda, plot_target_distribution, plot_feature_scatter


def main():
    # Load real dataset
    df = load_dataset(DATA_PATH)

    # Stage 1 schema audit print
    print('\n=== Dataset schema audit ===')
    print_schema(df)

    # Stage 2 target audit and cleaning
    clean_df, target_report = audit_target_y1(df)
    print('\n=== Target Y1 cleaning summary ===')
    for k, v in target_report.items():
        print(f'{k}: {v}')

    # Validate feature columns and physical relationships
    feature_list = validate_feature_columns(clean_df)
    checks = check_physical_feature_ranges(clean_df)

    # Multicollinearity relation check
    x2_minus_x3_2x4 = (clean_df['X2'] - clean_df['X3'] - 2 * clean_df['X4'])
    print('\n=== Physical relationship audit ===')
    print('X2 - X3 - 2*X4 max abs residual:', float(abs(x2_minus_x3_2x4).max()))
    print('X2 - X3 - 2*X4 mean residual:', float(x2_minus_x3_2x4.mean()))
    print('X2_X3_2X4 correlation:')
    print(clean_df[['X2', 'X3', 'X4']].corr().to_string())

    # EDA summary
    print('\n=== Basic EDA ===')
    basic_eda(clean_df)

    # Save figures
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    plot_target_distribution(clean_df, str(FIG_DIR), target='Y1')
    plot_target_distribution(clean_df, str(FIG_DIR), target='Y2')
    plot_feature_scatter(clean_df, str(FIG_DIR), target='Y1')
    plot_feature_scatter(clean_df, str(FIG_DIR), target='Y2')

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = clean_df[[c for c in clean_df.columns if c.startswith('X') or c in ['Y1', 'Y2']]].corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap with Y1 and Y2')
    plt.tight_layout()
    plt.savefig(str(FIG_DIR / 'correlation_heatmap.png'), dpi=150)
    plt.close()

    # Categorical boxplot examples Y1 by X5, X6, X7
    for col in ['X5', 'X6', 'X7']:
        plt.figure(figsize=(7, 4))
        if col in ['X5', 'X6', 'X7']:
            sns.boxplot(x=col, y='Y1', data=clean_df)
            plt.title(f'Y1 by {col}')
            plt.tight_layout()
            plt.savefig(str(FIG_DIR / f'y1_by_{col}.png'), dpi=150)
            plt.close()

    # Save a summary text report
    with open(OUT_DIR / 'stage2_report.txt', 'w', encoding='utf-8') as f:
        f.write('=== Stage 2: Cleaning + EDA Summary ===\n')
        f.write(f'Rows loaded: {df.shape[0]}\n')
        f.write(f'Columns loaded: {list(df.columns)}\n')
        f.write(f'Target Y1 valid rows: {target_report["target_numeric_rows"]}\n')
        f.write(f'Target Y1 invalid rows: {target_report["target_invalid_rows"]}\n')
        f.write(f'Target Y1 range: [{target_report["target_min"]}, {target_report["target_max"]}]\n')
        f.write(f'Leakage guard: Y2 must not enter X.\n')
        f.write(f'Feature list declared: {feature_list}\n')
        f.write(f'X2 - X3 - 2X4 max abs residual: {float(abs(x2_minus_x3_2x4).max())}\n')

    print('\n=== Stage 2 complete ===')
    print('Generated files:')
    print(FIG_DIR / 'y1_histogram.png')
    print(FIG_DIR / 'y1_boxplot.png')
    print(FIG_DIR / 'y2_histogram.png')
    print(FIG_DIR / 'y2_boxplot.png')
    print(FIG_DIR / 'correlation_heatmap.png')
    print(FIG_DIR / 'X1_vs_y1.png')
    print(FIG_DIR / 'X2_vs_y1.png')
    print(FIG_DIR / 'X5_vs_y1.png')
    print(OUT_DIR / 'stage2_report.txt')


if __name__ == '__main__':
    main()
