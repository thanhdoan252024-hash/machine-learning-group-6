import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

    for target in ['Y1', 'Y2']:
        print(f"\nTarget {target} summary:")
        print(df[target].describe())
        print("Skewness:", float(df[target].skew()))
        print("Kurtosis:", float(df[target].kurt()))

        print(f"\nCorrelation matrix with {target}:")
        corr = df[[c for c in df.columns if c.startswith('X') or c == target]].corr(numeric_only=True)
        print(corr[[target]].sort_values(target, ascending=False).to_string())

    print("\nFeature group cardinality summary:")
    for col in [f'X{i}' for i in range(1, 9)]:
        print(col, 'missing=', int(df[col].isna().sum()), 'unique=', int(df[col].nunique(dropna=True)))


def plot_target_distribution(df, outdir, target='Y1'):
    """
    Save histogram and boxplot supporting the EDA section.
    """
    plt.figure(figsize=(8, 4))
    target_name = 'Heating Load (Y1)' if target == 'Y1' else 'Cooling Load (Y2)'
    prefix = target.lower()
    plt.hist(df[target], bins=30, color='steelblue', edgecolor='black')
    plt.title(f'{target_name} Distribution')
    plt.xlabel(target_name)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{outdir}/{prefix}_histogram.png', dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.boxplot(df[target], vert=True)
    plt.title(f'{target_name} Boxplot')
    plt.ylabel(target_name)
    plt.tight_layout()
    plt.savefig(f'{outdir}/{prefix}_boxplot.png', dpi=150)
    plt.close()


def plot_feature_scatter(df, outdir, target='Y1'):
    """
    Save scatter plots for important architectural variables versus a target.
    """
    target_name = 'Heating Load (Y1)' if target == 'Y1' else 'Cooling Load (Y2)'
    prefix = target.lower()
    for col in ['X1', 'X2', 'X5']:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[col], df[target], alpha=0.65)
        plt.title(f'{col} vs {target_name}')
        plt.xlabel(col)
        plt.ylabel(target_name)
        plt.tight_layout()
        plt.savefig(f'{outdir}/{col}_vs_{prefix}.png', dpi=150)
        plt.close()
