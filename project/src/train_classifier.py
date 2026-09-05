"""
Training script for the XGBoost Classifier model.
Predicts behavior_class for transaction paths with overfitting/underfitting checks.

Classes:
    0 = Normal Flow
    1 = Rapid Movement
    2 = Fragmentation
    3 = Consolidation

Usage:
    python src/train_classifier.py
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CLASSIFIER_DATA, CLASSIFIER_MODEL_PATH, SCALER_CLF_PATH,
    FEATURE_COLUMNS, CLASSIFIER_TARGET, BEHAVIOR_LABELS,
    XGB_CLF_PARAM_GRID, CV_FOLDS, TEST_SIZE, RANDOM_STATE,
    MODELS_DIR, PLOTS_DIR,
)
from utils import (
    load_dataset, validate_data, plot_feature_importance,
    plot_confusion_matrix, plot_class_distribution,
    save_metadata, save_feature_list,
)


def train_classifier():
    """Full training pipeline for the XGBoost Classifier."""
    start_time = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # Step 1: Load & Validate Data
    # ──────────────────────────────────────────────────────────────
    df, feature_cols, target_col = load_dataset(CLASSIFIER_DATA, CLASSIFIER_TARGET)
    validation_report = validate_data(df, feature_cols, target_col)

    X = df[feature_cols].values
    y = df[target_col].values

    # ──────────────────────────────────────────────────────────────
    # Step 2: Train-Test Split
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Train-Test Split")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Training set: {X_train.shape[0]:,} samples")
    print(f"  Test set:     {X_test.shape[0]:,} samples")

    # ──────────────────────────────────────────────────────────────
    # Step 3: Feature Scaling
    # ──────────────────────────────────────────────────────────────
    print(f"\n  Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ──────────────────────────────────────────────────────────────
    # Step 4: GridSearchCV (XGBClassifier)
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GridSearchCV — XGBoost Classifier")
    print(f"{'='*60}")

    xgb_clf = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mlogloss"
    )

    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=XGB_CLF_PARAM_GRID,
        cv=CV_FOLDS,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    print(f"  Starting grid search with {CV_FOLDS}-fold CV...")
    print(f"  Parameter combinations: {np.prod([len(v) for v in XGB_CLF_PARAM_GRID.values()])}")
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"\n  ✓ Best Parameters: {grid_search.best_params_}")
    print(f"  ✓ Best CV F1 Score: {grid_search.best_score_:.6f}")

    # ──────────────────────────────────────────────────────────────
    # Step 5: Cross Validation Score
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Cross Validation (5-Fold)")
    print(f"{'='*60}")

    cv_scores = cross_val_score(
        best_model, X_train_scaled, y_train,
        cv=CV_FOLDS, scoring="f1_weighted", n_jobs=-1
    )
    print(f"  CV F1 Scores: {cv_scores}")
    print(f"  Mean CV F1:   {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")

    # ──────────────────────────────────────────────────────────────
    # Step 6: Evaluation & Overfitting/Underfitting Check
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Train vs Test Evaluation (Generalization Check)")
    print(f"{'='*60}")

    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    y_test_prob = best_model.predict_proba(X_test_scaled)

    train_f1 = f1_score(y_train, y_train_pred, average="weighted")
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average="weighted")
    recall = recall_score(y_test, y_test_pred, average="weighted")

    try:
        roc_auc = roc_auc_score(y_test, y_test_prob, multi_class="ovr", average="weighted")
    except Exception:
        roc_auc = None

    print(f"  Train F1: {train_f1:.6f}")
    print(f"  Test F1:  {test_f1:.6f}")
    print(f"  Train-Test Gap: {abs(train_f1 - test_f1):.6f}")
    print(f"  Accuracy:  {accuracy:.6f}")
    print(f"  Precision: {precision:.6f}")
    print(f"  Recall:    {recall:.6f}")
    if roc_auc is not None:
        print(f"  ROC-AUC:   {roc_auc:.6f}")

    if abs(train_f1 - test_f1) > 0.05:
        print("  ⚠️ Warning: Potential overfitting detected (Gap > 0.05)")
    else:
        print("  ✓ Optimal fit confirmed: Train and Test performance are closely aligned.")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")

    # Classification report
    class_names = [BEHAVIOR_LABELS[i] for i in sorted(BEHAVIOR_LABELS.keys())]
    report_str = classification_report(y_test, y_test_pred, target_names=class_names)
    print(f"\n  Classification Report:")
    print(report_str)

    # ──────────────────────────────────────────────────────────────
    # Step 7: Generate Plots
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating Visualizations")
    print(f"{'='*60}")

    # Feature importance
    plot_feature_importance(
        best_model.feature_importances_,
        feature_cols,
        "XGBoost Classifier — Feature Importance",
        os.path.join(PLOTS_DIR, "classifier_feature_importance.png"),
    )

    # Confusion matrix
    plot_confusion_matrix(
        cm, class_names,
        f"XGBoost Confusion Matrix (Accuracy={accuracy:.4f})",
        os.path.join(PLOTS_DIR, "classifier_confusion_matrix.png"),
    )

    # Class distribution (test set)
    plot_class_distribution(
        y_test, BEHAVIOR_LABELS,
        "Test Set — Behavior Class Distribution",
        os.path.join(PLOTS_DIR, "classifier_class_distribution.png"),
    )

    # ──────────────────────────────────────────────────────────────
    # Step 8: Save Model & Artifacts
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Saving Model & Artifacts")
    print(f"{'='*60}")

    joblib.dump(best_model, CLASSIFIER_MODEL_PATH)
    print(f"  ✓ Model saved: {CLASSIFIER_MODEL_PATH}")

    joblib.dump(scaler, SCALER_CLF_PATH)
    print(f"  ✓ Scaler saved: {SCALER_CLF_PATH}")

    # ──────────────────────────────────────────────────────────────
    # Step 9: Save Metadata
    # ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    metadata = {
        "model_type": "XGBClassifier",
        "task": "classification",
        "target": CLASSIFIER_TARGET,
        "classes": {str(k): v for k, v in BEHAVIOR_LABELS.items()},
        "features": feature_cols,
        "n_features": len(feature_cols),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "best_params": grid_search.best_params_,
        "metrics": {
            "accuracy": round(accuracy, 6),
            "precision_weighted": round(precision, 6),
            "recall_weighted": round(recall, 6),
            "f1_weighted": round(test_f1, 6),
            "f1_train": round(train_f1, 6),
            "roc_auc_weighted": round(roc_auc, 6) if roc_auc else None,
            "CV_F1_mean": round(cv_scores.mean(), 6),
            "CV_F1_std": round(cv_scores.std(), 6),
        },
        "feature_importances": {
            col: round(float(imp), 6)
            for col, imp in zip(feature_cols, best_model.feature_importances_)
        },
        "classification_report": report_str,
        "training_time_seconds": round(elapsed, 2),
        "validation_report": {
            "total_missing": validation_report["total_missing"],
            "duplicates": validation_report["duplicates"],
        },
    }

    # Save as separate metadata file for classifier
    clf_metadata_path = os.path.join(MODELS_DIR, "classifier_metadata.json")
    with open(clf_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  ✓ Metadata saved: {clf_metadata_path}")

    # ──────────────────────────────────────────────────────────────
    # Final Summary
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("XGBOOST CLASSIFIER TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Model:    {CLASSIFIER_MODEL_PATH}")
    print(f"  Accuracy: {accuracy:.6f}")
    print(f"  F1:       {test_f1:.6f}")
    print(f"  Time:     {elapsed:.1f}s")
    print()

    return best_model, scaler, metadata


if __name__ == "__main__":
    train_classifier()
