import pandas as pd
from pathlib import Path


def load_dataset(path):
    """
    Read the Energy Efficiency dataset from CSV and validate that the file exists.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    return df


def print_schema(df):
    """
    Print a human-readable audit summary for Dataset Audit Stage 1.
    """
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Dtypes:")
    print(df.dtypes)
    print("Missing values:")
    print(df.isna().sum())
    print("Unique values:")
    print(df.nunique(dropna=True))
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("Sample rows:")
    print(df.head())
