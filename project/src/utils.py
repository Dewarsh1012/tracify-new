"""
Utility functions for the Blockchain Forensics ML Pipeline.
Provides data loading, validation, risk level logic, and plotting helpers.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_COLUMNS, RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD,
    BEHAVIOR_LABELS, PLOTS_DIR,
)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("deep")


# ══════════════════════════════════════════════════════════════════════
# Data Loading & Validation
# ══════════════════════════════════════════════════════════════════════

def load_dataset(filepath: str, target_col: str) -> tuple:
    """
    Load CSV dataset and return (DataFrame, feature_names, target_name).
    Performs basic validation checks.
    """
    print(f"\n{'='*60}")
    print(f"Loading dataset: {os.path.basename(filepath)}")
    print(f"{'='*60}")

    df = pd.read_csv(filepath)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    return df, FEATURE_COLUMNS, target_col


def validate_data(df: pd.DataFrame, feature_cols: list, target_col: str) -> dict:
    """
    Comprehensive data validation: missing values, duplicates, outliers, stats.
    Returns a validation report dict.
    """
    report = {}

    # 1. Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    report["missing_values"] = missing.to_dict()
    report["total_missing"] = int(total_missing)
    print(f"\n  [1/5] Missing Values: {total_missing}")
    if total_missing > 0:
        print(f"    {missing[missing > 0].to_dict()}")
    else:
        print(f"    ✓ No missing values detected")

    # 2. Duplicates
    n_duplicates = df.duplicated().sum()
    report["duplicates"] = int(n_duplicates)
    print(f"  [2/5] Duplicates: {n_duplicates}")
    if n_duplicates > 0:
        print(f"    ⚠ Found {n_duplicates} duplicate rows")
    else:
        print(f"    ✓ No duplicate rows")

    # 3. Outlier summary (IQR method)
    print(f"  [3/5] Outlier Summary (IQR method):")
    outlier_counts = {}
    for col in feature_cols:
        if col in df.columns and df[col].dtype in [np.float64, np.int64, float, int]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            outlier_counts[col] = int(n_outliers)
            if n_outliers > 0:
                print(f"    {col}: {n_outliers} outliers "
                      f"(range: [{lower:.2f}, {upper:.2f}])")
    report["outliers"] = outlier_counts

    # 4. Feature statistics
    print(f"  [4/5] Feature Statistics:")
    stats = df[feature_cols].describe().round(4)
    print(stats.to_string())
    report["feature_stats"] = stats.to_dict()

    # 5. Target statistics
    print(f"  [5/5] Target Variable ({target_col}):")
    if df[target_col].dtype in [np.float64, float]:
        print(f"    Range: [{df[target_col].min():.2f}, {df[target_col].max():.2f}]")
        print(f"    Mean: {df[target_col].mean():.2f}, Std: {df[target_col].std():.2f}")
    else:
        vc = df[target_col].value_counts().sort_index()
        for cls, count in vc.items():
            label = BEHAVIOR_LABELS.get(cls, str(cls))
            print(f"    Class {cls} ({label}): {count:,} ({count/len(df)*100:.1f}%)")
    report["target_stats"] = df[target_col].describe().to_dict()

    return report


# ══════════════════════════════════════════════════════════════════════
# Risk Level Logic
# ══════════════════════════════════════════════════════════════════════

def get_risk_level(relevance_score: float) -> str:
    """
    Determine risk level from relevance score.

    Risk Levels:
        - HIGH:   relevance_score >= 80
        - MEDIUM: relevance_score >= 50
        - LOW:    relevance_score < 50
    """
    if relevance_score >= RISK_HIGH_THRESHOLD:
        return "HIGH"
    elif relevance_score >= RISK_MEDIUM_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


def get_behavior_label(behavior_class: int) -> str:
    """Map behavior class integer to human-readable label."""
    return BEHAVIOR_LABELS.get(int(behavior_class), "Unknown")


# ══════════════════════════════════════════════════════════════════════
# Plotting Helpers
# ══════════════════════════════════════════════════════════════════════

def plot_feature_importance(importances, feature_names, title, filepath):
    """Plot horizontal bar chart of feature importances."""
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    ax.barh(range(len(feature_names)),
            importances[indices],
            color=colors[indices],
            edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    # Add value labels
    for i, idx in enumerate(indices):
        ax.text(importances[idx] + 0.002, i, f"{importances[idx]:.4f}",
                va="center", fontsize=9, color="#333")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_actual_vs_predicted(y_actual, y_predicted, title, filepath):
    """Scatter plot of actual vs predicted values for regression."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_actual, y_predicted, alpha=0.15, s=8, color="#4C72B0", rasterized=True)
    ax.plot([0, 100], [0, 100], "r--", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Relevance Score", fontsize=12)
    ax.set_ylabel("Predicted Relevance Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=11)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_residual_distribution(y_actual, y_predicted, title, filepath):
    """Histogram of residuals for regression model."""
    residuals = y_actual - y_predicted
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=60, color="#55A868", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Residual")
    ax.set_xlabel("Residual (Actual − Predicted)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    # Stats annotation
    mean_r = np.mean(residuals)
    std_r = np.std(residuals)
    ax.text(0.72, 0.92, f"Mean: {mean_r:.3f}\nStd: {std_r:.3f}",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_confusion_matrix(cm, class_labels, title, filepath):
    """Plot annotated confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels,
                ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def plot_class_distribution(y, class_labels_map, title, filepath):
    """Bar chart of class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    labels = [class_labels_map.get(int(u), str(u)) for u in unique]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, counts, color=colors[:len(labels)], edgecolor="white")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{count:,}\n({count/sum(counts)*100:.1f}%)",
                ha="center", fontsize=10)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


def save_metadata(metadata: dict, filepath: str):
    """Save model metadata as JSON."""
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved metadata: {filepath}")


def save_feature_list(features: list, filepath: str):
    """Save feature list as JSON."""
    with open(filepath, "w") as f:
        json.dump({"features": features, "count": len(features)}, f, indent=2)
    print(f"  Saved feature list: {filepath}")
