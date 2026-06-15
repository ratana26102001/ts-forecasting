"""
visualization.py
Plotting helpers. All figures saved to outputs/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", palette="muted")
Path("outputs").mkdir(exist_ok=True)

COLORS = {
    "SARIMA":  "#ef4444",
    "XGBoost": "#f59e0b",
    "LSTM":    "#10b981",
}


def plot_raw_series(df: pd.DataFrame, date_col: str = "ds", target_col: str = "y"):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df[date_col], df[target_col], color="#2563eb", linewidth=0.7, alpha=0.85)
    ax.set_title("Raw Time Series", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Value")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.tight_layout()
    plt.savefig("outputs/01_raw_series.png", dpi=150)
    plt.close()
    print("Saved → outputs/01_raw_series.png")


def plot_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    date_col: str = "ds",
    target_col: str = "y",
):
    n_context = max(100, len(train) // 10)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        train[date_col].tail(n_context), train[target_col].tail(n_context),
        color="#9ca3af", linewidth=1, label="Train (context)", alpha=0.7,
    )
    ax.plot(test[date_col], test[target_col], color="#1e293b", linewidth=1.5, label="Actual")
    for name, preds in predictions.items():
        ax.plot(test[date_col], preds, linewidth=1.5, linestyle="--",
                label=name, color=COLORS.get(name, "#8b5cf6"))
    ax.set_title("Forecast Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Value")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.tight_layout()
    plt.savefig("outputs/02_forecast_comparison.png", dpi=150)
    plt.close()
    print("Saved → outputs/02_forecast_comparison.png")


def plot_metrics(metrics_df: pd.DataFrame):
    cols = ["MAE", "RMSE", "MAPE (%)"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, cols):
        vals = metrics_df[col]
        bars = ax.bar(vals.index, vals,
                      color=[COLORS.get(m, "#6366f1") for m in vals.index],
                      edgecolor="white", linewidth=0.5)
        ax.set_title(col, fontweight="bold")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.suptitle("Model Performance Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/03_metrics_comparison.png", dpi=150)
    plt.close()
    print("Saved → outputs/03_metrics_comparison.png")


def plot_residuals(
    test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    date_col: str = "ds",
    target_col: str = "y",
):
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, preds) in zip(axes, predictions.items()):
        residuals = np.asarray(test[target_col].values) - np.asarray(preds)
        ax.plot(test[date_col], residuals,
                linewidth=0.7, color=COLORS.get(name, "#8b5cf6"), alpha=0.85)
        ax.axhline(0, linestyle="--", color="black", linewidth=1)
        ax.set_title(f"{name} Residuals", fontweight="bold")
        ax.set_xlabel("Date"); ax.set_ylabel("Residual")
    plt.tight_layout()
    plt.savefig("outputs/04_residuals.png", dpi=150)
    plt.close()
    print("Saved → outputs/04_residuals.png")
