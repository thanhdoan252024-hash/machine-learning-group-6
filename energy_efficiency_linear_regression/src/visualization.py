import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns



def plot_training_loss(loss_history, output_path):
    """
    Plot training loss curve with iteration x-axis and MSE loss y-axis.
    """
    iterations = np.arange(1, len(loss_history) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, loss_history, color='blue', linewidth=1.5)
    plt.title('Training Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred, output_path):
    """
    Actual vs predicted plot with y=x line.
    """
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', linewidth=1.0)
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual Heating Load (Y1)')
    plt.ylabel('Predicted Heating Load (Y1)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residuals(pred, y_true, output_path):
    """
    Residual plot: Y predicted vs residuals, y=0 line.
    """
    residuals = y_true - pred
    plt.figure(figsize=(7, 5))
    plt.scatter(pred, residuals, alpha=0.5, s=20)
    plt.axhline(0, color='black', linewidth=1.0)
    plt.title('Residual Plot')
    plt.xlabel('Predicted Heating Load (Y1)')
    plt.ylabel('Residual (y - y_hat)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residual_distribution(residuals, output_path):
    """
    Plot residual distribution by histogram and KDE.
    """
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title('Residual Distribution')
    plt.xlabel('Residual (y - y_hat)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
