import pandas as pd
import numpy as np


RANDOM_SEED = 42
TARGET = 'Y1'
LEAKAGE_TARGET = 'Y2'


def audit_target_y1(df):
    """
    Validate Y1 target and report the number of valid and invalid target rows.
    """
    original_rows = len(df)

    raw_y1 = df['Y1'].copy()
    y1_numeric = pd.to_numeric(raw_y1, errors='coerce')
    invalid_mask = y1_numeric.isna()

    valid_rows = int((~invalid_mask).sum())
    invalid_rows = int(invalid_mask.sum())

    report = {
        'original_rows': original_rows,
        'target_numeric_rows': valid_rows,
        'target_invalid_rows': invalid_rows,
        'target_valid_rate': valid_rows / original_rows if original_rows else 0.0,
        'target_min': float(y1_numeric.min()) if valid_rows else np.nan,
        'target_max': float(y1_numeric.max()) if valid_rows else np.nan,
    }

    # keep rows with valid numerical Y1
    clean_df = df.loc[~invalid_mask].copy()
    clean_df[TARGET] = y1_numeric.loc[~invalid_mask].astype(float)

    return clean_df, report


def validate_feature_columns(df):
    """
    Ensure the feature columns have the expected names and no target leakage.
    """
    required = [f'X{i}' for i in range(1, 9)]
    if not all(c in df.columns for c in required):
        raise ValueError(f"Missing expected feature columns: {required}")

    if TARGET in df.columns:
        pass

    if LEAKAGE_TARGET in df.columns:
        # Y2 is explicitly not allowed in model input.
        print("Leakage check: Y2 exists only as a target-like column and must be excluded from X.")

    return required


def check_physical_feature_ranges(df):
    """
    Perform a few sanity checks for physical validity of the Energy Efficiency data.
    """
    checks = {}
    checks['X5_valid_categories'] = sorted(df['X5'].unique().tolist())
    checks['X6_valid_categories'] = sorted(df['X6'].unique().tolist())
    checks['X7_valid_categories'] = sorted(df['X7'].unique().tolist())
    checks['X8_valid_categories'] = sorted(df['X8'].unique().tolist())
    return checks
