import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'energy_efficiency_building_heating_cooling_load_dataset.csv'
OUT_DIR = PROJECT_ROOT / 'outputs'
TABLE_DIR = OUT_DIR / 'tables'

sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from preprocessing import preprocess_pipeline, RANDOM_SEED
from data_loader import load_dataset


def main():
    df = load_dataset(DATA_PATH)

    # Remove Y2 from the matrix to enforce clean feature pipeline
    if 'Y2' in df.columns:
        df = df.drop(columns=['Y2'])

    X_train, X_test, y_train, y_test, medians, scale_info = preprocess_pipeline(
        df,
        target_col='Y1',
        test_size=0.2,
        random_seed=RANDOM_SEED,
    )

    # Save output summary
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'stage3_report.txt', 'w', encoding='utf-8') as f:
        f.write('=== Stage 3: Feature Engineering + Preprocessing ===\n')
        f.write(f'Dataset rows loaded: {df.shape[0]}\n')
        f.write(f'Dataset columns loaded: {list(df.columns)}\n')
        f.write(f'Train rows: {len(y_train)}\n')
        f.write(f'Test rows: {len(y_test)}\n')
        f.write(f'Seed: {RANDOM_SEED}\n')
        f.write(f'Train features: {X_train.shape[1]}\n')
        f.write(f'Test features: {X_test.shape[1]}\n')
        f.write(f'Median imputation values: {medians}\n')
        f.write(f'Scale info: {scale_info}\n')
        f.write(f'Y2 leak guard: removed from X matrix before train/test split.\n')
        f.write(f'Categorical encoding: X6 and X8 one-hot with drop_first=True.\n')
        f.write(f'Numerical scale: z-score fit on train then applied to test.\n')

    print('=== Stage 3 preprocessing complete ===')
    print(f'Train shape: {X_train.shape}, y_train: {y_train.shape}')
    print(f'Test shape: {X_test.shape}, y_test: {y_test.shape}')
    print(f'Feature count: {X_train.shape[1]}')
    print(f'Medians used: {medians}')
    print('Report written:', OUT_DIR / 'stage3_report.txt')


if __name__ == '__main__':
    main()
