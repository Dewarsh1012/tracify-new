"""
Explainability module for the Blockchain Forensics ML Pipeline.
Generates feature importance rankings, SHAP analysis, and global explanation reports for XGBoost models.

Usage:
    python src/explainability.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    REGRESSOR_MODEL_PATH, CLASSIFIER_MODEL_PATH,
    SCALER_REG_PATH, SCALER_CLF_PATH,
    REGRESSOR_DATA, CLASSIFIER_DATA,
    FEATURE_COLUMNS, REGRESSOR_TARGET, CLASSIFIER_TARGET,
    BEHAVIOR_LABELS, PLOTS_DIR, MODELS_DIR,
)

plt.style.use("seaborn-v0_8-darkgrid")


def generate_explanation_report():
    """Generate comprehensive explainability report for both XGBoost models."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("EXPLAINABILITY REPORT — XGBOOST MODELS")
    print("=" * 60)

    # Load models
    reg_model = joblib.load(REGRESSOR_MODEL_PATH)
    clf_model = joblib.load(CLASSIFIER_MODEL_PATH)

    # ── Feature Importance Ranking ──
    print("\n── Feature Importance Ranking ──")

    reg_imp = reg_model.feature_importances_
    clf_imp = clf_model.feature_importances_

    print("\n  Top 10 Features — XGBoost Regressor:")
    reg_sorted = sorted(zip(FEATURE_COLUMNS, reg_imp), key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(reg_sorted[:10], 1):
        bar = "█" * int(imp * 100)
        print(f"    {i:2d}. {feat:<25s} {imp:.4f}  {bar}")

    print("\n  Top 10 Features — XGBoost Classifier:")
    clf_sorted = sorted(zip(FEATURE_COLUMNS, clf_imp), key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(clf_sorted[:10], 1):
        bar = "█" * int(imp * 100)
        print(f"    {i:2d}. {feat:<25s} {imp:.4f}  {bar}")

    # ── Combined importance comparison plot ──
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(FEATURE_COLUMNS))
    width = 0.35

    reg_order = np.argsort(reg_imp)[::-1]
    ordered_features = [FEATURE_COLUMNS[i] for i in reg_order]
    reg_vals = reg_imp[reg_order]
    clf_vals = clf_imp[reg_order]

    bars1 = ax.barh(x - width / 2, reg_vals, width, label="XGBoost Regressor", color="#55A868", edgecolor="white")
    bars2 = ax.barh(x + width / 2, clf_vals, width, label="XGBoost Classifier", color="#4C72B0", edgecolor="white")

    ax.set_yticks(x)
    ax.set_yticklabels(ordered_features, fontsize=11)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Feature Importance Comparison — XGBoost Regressor vs Classifier",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "explainability_importance_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {os.path.join(PLOTS_DIR, 'explainability_importance_comparison.png')}")

    # ── SHAP Analysis ──
    print("\n── SHAP Analysis ──")
    shap_available = False
    try:
        import shap
        shap_available = True
        print("  SHAP library found. Generating SHAP analysis...")

        # Load data samples for SHAP
        reg_df = pd.read_csv(REGRESSOR_DATA)
        X_sample = reg_df[FEATURE_COLUMNS].sample(n=min(1000, len(reg_df)), random_state=42)
        scaler_reg = joblib.load(SCALER_REG_PATH)
        X_sample_scaled = scaler_reg.transform(X_sample.values)

        # SHAP for regressor
        explainer_reg = shap.TreeExplainer(reg_model)
        shap_values_reg = explainer_reg.shap_values(X_sample_scaled)

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values_reg, X_sample, feature_names=FEATURE_COLUMNS,
                          show=False)
        plt.title("SHAP Summary — XGBoost Regressor", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "shap_regressor_summary.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {os.path.join(PLOTS_DIR, 'shap_regressor_summary.png')}")

        # SHAP for classifier
        clf_df = pd.read_csv(CLASSIFIER_DATA)
        X_sample_clf = clf_df[FEATURE_COLUMNS].sample(n=min(1000, len(clf_df)), random_state=42)
        scaler_clf = joblib.load(SCALER_CLF_PATH)
        X_sample_clf_scaled = scaler_clf.transform(X_sample_clf.values)

        explainer_clf = shap.TreeExplainer(clf_model)
        shap_values_clf = explainer_clf.shap_values(X_sample_clf_scaled)

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values_clf[0] if isinstance(shap_values_clf, list) else shap_values_clf,
                          X_sample_clf, feature_names=FEATURE_COLUMNS, show=False)
        plt.title("SHAP Summary — XGBoost Classifier (Class 0: Normal Flow)",
                  fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "shap_classifier_summary.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {os.path.join(PLOTS_DIR, 'shap_classifier_summary.png')}")

    except Exception as e:
        print(f"  ⚠️ SHAP generation note: {e}")

    # ── Global Explanation Report ──
    print(f"\n{'='*60}")
    print("GLOBAL EXPLANATION REPORT")
    print(f"{'='*60}")

    report = {
        "regressor": {
            "model_type": "XGBRegressor",
            "target": "relevance_score (0-100)",
            "top_features": [
                {"rank": i + 1, "feature": feat, "importance": round(float(imp), 6)}
                for i, (feat, imp) in enumerate(reg_sorted)
            ],
            "interpretation": {
                reg_sorted[0][0]: f"Most important feature ({reg_sorted[0][1]:.4f}). "
                                  "Strongly drives relevance predictions.",
                reg_sorted[1][0]: f"Second most important ({reg_sorted[1][1]:.4f}). "
                                  "Key contributing factor.",
                reg_sorted[2][0]: f"Third most important ({reg_sorted[2][1]:.4f}).",
            },
        },
        "classifier": {
            "model_type": "XGBClassifier",
            "target": "behavior_class (0=Normal, 1=Rapid, 2=Fragment, 3=Consolidate)",
            "top_features": [
                {"rank": i + 1, "feature": feat, "importance": round(float(imp), 6)}
                for i, (feat, imp) in enumerate(clf_sorted)
            ],
            "interpretation": {
                clf_sorted[0][0]: f"Most discriminative feature ({clf_sorted[0][1]:.4f}). "
                                  "Primary driver of behavior classification.",
                clf_sorted[1][0]: f"Second most discriminative ({clf_sorted[1][1]:.4f}).",
                clf_sorted[2][0]: f"Third most discriminative ({clf_sorted[2][1]:.4f}).",
            },
        },
        "shap_available": shap_available,
    }

    # Save report
    report_path = os.path.join(MODELS_DIR, "global_explanation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {report_path}")

    # Print summary
    print("\n  Regressor — Key Insights:")
    for feat, imp in reg_sorted[:5]:
        print(f"    • {feat}: {imp:.4f}")

    print("\n  Classifier — Key Insights:")
    for feat, imp in clf_sorted[:5]:
        print(f"    • {feat}: {imp:.4f}")

    print(f"\n{'='*60}")
    print("Explainability report complete ✓")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    generate_explanation_report()
