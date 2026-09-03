"""
Training script for the Random Forest Classifier model.
Predicts behavior_class for transaction paths.

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

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
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
    RF_PARAM_GRID, CV_FOLDS, TEST_SIZE, RANDOM_STATE,
    MODELS_DIR, PLOTS_DIR,
)
from utils import (
    load_dataset, validate_data, plot_feature_importance,
    plot_confusion_matrix, plot_class_distribution,
    save_metadata, save_feature_list,
)


def train_classifier():
    """Full training pipeline for the Random Forest Classifier."""
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
    # Step 4: GridSearchCV
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GridSearchCV — Random Forest Classifier")
    print(f"{'='*60}")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [15, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1],
        "max_features": ["sqrt"],
    }

    rf = RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
    )

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    print(f"  Starting grid search with {CV_FOLDS}-fold CV...")
    print(f"  Parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")
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
    # Step 6: Evaluation on Test Set
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Test Set Evaluation")
    print(f"{'='*60}")

    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # ROC-AUC (one-vs-rest)
    try:
        roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        roc_auc = None

    print(f"  Accuracy:  {accuracy:.6f}")
    print(f"  Precision: {precision:.6f}")
    print(f"  Recall:    {recall:.6f}")
    print(f"  F1 Score:  {f1:.6f}")
    if roc_auc is not None:
        print(f"  ROC-AUC:   {roc_auc:.6f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")

    # Classification report
    class_names = [BEHAVIOR_LABELS[i] for i in sorted(BEHAVIOR_LABELS.keys())]
    report_str = classification_report(y_test, y_pred, target_names=class_names)
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
        "Random Forest Classifier — Feature Importance",
        os.path.join(PLOTS_DIR, "classifier_feature_importance.png"),
    )

    # Confusion matrix
    plot_confusion_matrix(
        cm, class_names,
        f"Confusion Matrix (Accuracy={accuracy:.4f})",
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
        "model_type": "RandomForestClassifier",
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
            "f1_weighted": round(f1, 6),
            "roc_auc_weighted": round(roc_auc, 6) if roc_auc else None,
            "CV_F1_mean": round(cv_scores.mean(), 6),
            "CV_F1_std": round(cv_scores.std(), 6),
        },
        "feature_importances": {
            col: round(imp, 6)
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
    print("CLASSIFIER TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Model:    {CLASSIFIER_MODEL_PATH}")
    print(f"  Accuracy: {accuracy:.6f}")
    print(f"  F1:       {f1:.6f}")
    print(f"  Time:     {elapsed:.1f}s")
    print()

    return best_model, scaler, metadata


if __name__ == "__main__":
    train_classifier()
