import numpy as np
import pandas as pd


NUMERIC_COLS = ['X1', 'X3', 'X4', 'X5', 'X7']
CATEGORICAL_COLS = ['X6', 'X8']
TARGET_COL = 'Y1'
LEAKAGE_COL = 'Y2'
RANDOM_SEED = 42


def make_kfold_splits(df, target_col='Y1', n_splits=5, random_seed=42):
    """
    Create deterministic k-fold validation partitions while keeping the
    target-leakage guard active. For each target, the opposite target is
    removed before the split and both train/test frames remain consistent.
    """
    if n_splits < 2:
        raise ValueError('n_splits must be >= 2')

    work_df = df.copy()
    if target_col == 'Y1' and 'Y2' in work_df.columns:
        work_df = work_df.drop(columns=['Y2'])
    elif target_col == 'Y2' and 'Y1' in work_df.columns:
        work_df = work_df.drop(columns=['Y1'])

    rng = np.random.RandomState(random_seed)
    idx = rng.permutation(np.arange(len(work_df)))
    folds = np.array_split(idx, n_splits)

    splits = []
    for fold in folds:
        test_idx = set(fold.tolist())
        train_idx = [i for i in range(len(work_df)) if i not in test_idx]
        train_df = work_df.iloc[train_idx].reset_index(drop=True)
        test_df = work_df.iloc[list(test_idx)].reset_index(drop=True)
        splits.append((train_df, test_df))

    return splits


def custom_train_test_split(df, target_col='Y1', test_size=0.2, random_seed=42):
    """
    Split data before preprocessing using a deterministic NumPy permutation.
    Returns train/test frames and target vectors.
    """
    rng = np.random.RandomState(random_seed)
    n = len(df)
    n_test = int(round(test_size * n))
    shuffled_idx = rng.permutation(np.arange(n))
    test_idx = shuffled_idx[:n_test]
    train_idx = shuffled_idx[n_test:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    y_train = df_train[target_col].astype(float).to_numpy()
    y_test = df_test[target_col].astype(float).to_numpy()

    return df_train, df_test, y_train, y_test


def fit_numeric_medians(train_df, numeric_cols):
    """
    Fit numeric missing imputer on TRAIN ONLY.
    """
    medians = {}
    for col in numeric_cols:
        medians[col] = float(train_df[col].median())
    return medians


def apply_numeric_medians(df, medians, numeric_cols):
    """
    Apply median imputation on any data frame consistent with the column names.
    """
    out = df.copy()
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors='coerce')
        out[col] = out[col].fillna(medians[col])
    return out


def onehot_encode_train_test(train_df, test_df, categorical_cols):
    """
    One-hot encode categorical columns using train categories only. Align test columns
    to training columns and fill unseen categories with zero.
    """
    train_encoded = []
    test_encoded = []

    for col in categorical_cols:
        train_dummies = pd.get_dummies(train_df[col], prefix=col, drop_first=True, dtype=np.float64)
        test_dummies = pd.get_dummies(test_df[col], prefix=col, drop_first=True, dtype=np.float64)

        # Align columns to the training set, ensuring same order
        train_dummies_cols = train_dummies.columns.tolist()
        test_dummies = test_dummies.reindex(columns=train_dummies_cols, fill_value=0.0)
        train_dummies = train_dummies.reindex(columns=train_dummies_cols, fill_value=0.0)

        train_encoded.append(train_dummies)
        test_encoded.append(test_dummies)

    # Concatenate all encoded categorical features
    X_train_cat = pd.concat(train_encoded, axis=1)
    X_test_cat = pd.concat(test_encoded, axis=1)
    return X_train_cat, X_test_cat


def preprocess_pipeline(df, target_col='Y1', test_size=0.2, random_seed=42, train_df=None, test_df=None):
    """
    End-to-end preprocessing pipeline respecting train-only fit logic.
    Returns processed train/test frames and y arrays.

    Supports either a normal split or an externally supplied train/test frame pair,
    used by the cross-validation workflow and by the model evaluation stage.

    Leakage-safe rule:
      - when target_col='Y1', drop Y2 from the feature frame.
      - when target_col='Y2', drop Y1 from the feature frame.
    """
    if target_col == 'Y1' and 'Y2' in df.columns:
        df = df.drop(columns=['Y2'])
    elif target_col == 'Y2' and 'Y1' in df.columns:
        df = df.drop(columns=['Y1'])

    # Option A: inherit externally supplied train/test partitions.
    if train_df is not None and test_df is not None:
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df = train_df.drop(columns=[c for c in ['Y1', 'Y2'] if c != target_col and c in train_df.columns], errors='ignore')
        test_df = test_df.drop(columns=[c for c in ['Y1', 'Y2'] if c != target_col and c in test_df.columns], errors='ignore')
        y_train = train_df[target_col].astype(float).to_numpy()
        y_test = test_df[target_col].astype(float).to_numpy()
        train_df = train_df.drop(columns=[target_col])
        test_df = test_df.drop(columns=[target_col])
    else:
        # Option B: split first
        train_df, test_df, y_train, y_test = custom_train_test_split(
            df=df,
            target_col=target_col,
            test_size=test_size,
            random_seed=random_seed,
        )

    # Fit medians on train only
    medians = fit_numeric_medians(train_df, NUMERIC_COLS)
    train_df_num = apply_numeric_medians(train_df, medians, NUMERIC_COLS)
    test_df_num = apply_numeric_medians(test_df, medians, NUMERIC_COLS)

    # Encode categories from train and align test
    X_train_cat, X_test_cat = onehot_encode_train_test(train_df_num, test_df_num, CATEGORICAL_COLS)

    # Numerical features remain numeric
    X_train_num = train_df_num[NUMERIC_COLS].reset_index(drop=True)
    X_test_num = test_df_num[NUMERIC_COLS].reset_index(drop=True)

    # Merge columns in consistent order
    X_train = pd.concat([X_train_num.reset_index(drop=True), X_train_cat.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test_num.reset_index(drop=True), X_test_cat.reset_index(drop=True)], axis=1)

    # Sort columns for reproducible alignment
    X_train = X_train.reindex(sorted(X_train.columns), axis=1)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    # Scale train/test numerically only using train mean/std (z-score) for numeric columns after imputation
    numeric_scale_info = {}
    for col in NUMERIC_COLS:
        mu = X_train[col].mean()
        sigma = X_train[col].std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            sigma = 1.0
        X_train[col] = (X_train[col] - mu) / sigma
        X_test[col] = (X_test[col] - mu) / sigma
        numeric_scale_info[col] = {'mu': mu, 'sigma': sigma}

    # Keep converted design matrix columns in consistent order
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    return X_train, X_test, y_train, y_test, medians, numeric_scale_info
