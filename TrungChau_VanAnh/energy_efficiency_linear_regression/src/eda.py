import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


TARGET = 'Y1'


def basic_eda(df):
    """
    Create a compact EDA report that matches the requested stage 2 requirements.
    """
    print("Dataset Overview")
    print("Rows:", df.shape[0], "Columns:", df.shape[1])
    print("Dtypes:")
    print(df.dtypes)
    print("Missing values:")
    print(df.isna().sum())
    print("Duplicate rows:", int(df.duplicated().sum()))

    print("\nCategorical cardinality:")
    for col in [c for c in df.columns if c in ['X6', 'X8']]:
        print(col, 'unique=', df[col].nunique(dropna=True))

    print("\nTarget Y1 summary:")
    print(df[TARGET].describe())
    print("Skewness:", float(df[TARGET].skew()))
    print("Kurtosis:", float(df[TARGET].kurt()))

    print("\nCorrelation matrix with Y1:")
    corr = df[[c for c in df.columns if c.startswith('X') or c == TARGET]].corr(numeric_only=True)
    print(corr[[TARGET]].sort_values(TARGET, ascending=False).to_string())

    print("\nFeature group cardinality summary:")
    for col in [f'X{i}' for i in range(1, 9)]:
        print(col, 'missing=', int(df[col].isna().sum()), 'unique=', int(df[col].nunique(dropna=True)))


def plot_target_distribution(df, outdir):
    """
    Save the Y1 histogram and boxplot supporting the EDA section.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(df[TARGET], bins=30, color='steelblue', edgecolor='black')
    plt.title('Y1 Heating Load Distribution')
    plt.xlabel('Y1 Heating Load')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{outdir}/y1_histogram.png', dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[TARGET])
    plt.title('Y1 Heating Load Boxplot')
    plt.ylabel('Y1 Heating Load')
    plt.tight_layout()
    plt.savefig(f'{outdir}/y1_boxplot.png', dpi=150)
    plt.close()


def plot_feature_scatter(df, outdir):
    """
    Save scatter plots for important architectural variables versus Y1.
    """
    for col in ['X1', 'X2', 'X5']:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[col], df[TARGET], alpha=0.65)
        plt.title(f'{col} vs Y1')
        plt.xlabel(col)
        plt.ylabel('Y1 Heating Load')
        plt.tight_layout()
        plt.savefig(f'{outdir}/{col}_vs_y1.png', dpi=150)
        plt.close()
