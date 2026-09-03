"""
Training script for the Random Forest Regressor model.
Predicts relevance_score (0-100) for transaction paths.

Usage:
    python src/train_regressor.py
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    REGRESSOR_DATA, REGRESSOR_MODEL_PATH, SCALER_REG_PATH,
    METADATA_PATH, FEATURE_LIST_PATH, FEATURE_COLUMNS,
    REGRESSOR_TARGET, RF_PARAM_GRID, CV_FOLDS,
    TEST_SIZE, RANDOM_STATE, MODELS_DIR, PLOTS_DIR,
)
from utils import (
    load_dataset, validate_data, plot_feature_importance,
    plot_actual_vs_predicted, plot_residual_distribution,
    save_metadata, save_feature_list,
)


def train_regressor():
    """Full training pipeline for the Random Forest Regressor."""
    start_time = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # Step 1: Load & Validate Data
    # ──────────────────────────────────────────────────────────────
    df, feature_cols, target_col = load_dataset(REGRESSOR_DATA, REGRESSOR_TARGET)
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
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
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
    print("GridSearchCV — Random Forest Regressor")
    print(f"{'='*60}")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [15, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1],
        "max_features": ["sqrt"],
    }

    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    print(f"  Starting grid search with {CV_FOLDS}-fold CV...")
    print(f"  Parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"\n  ✓ Best Parameters: {grid_search.best_params_}")
    print(f"  ✓ Best CV R² Score: {grid_search.best_score_:.6f}")

    # ──────────────────────────────────────────────────────────────
    # Step 5: Cross Validation Score
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Cross Validation (5-Fold)")
    print(f"{'='*60}")

    cv_scores = cross_val_score(
        best_model, X_train_scaled, y_train,
        cv=CV_FOLDS, scoring="r2", n_jobs=-1
    )
    print(f"  CV R² Scores: {cv_scores}")
    print(f"  Mean CV R²:   {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")

    # ──────────────────────────────────────────────────────────────
    # Step 6: Evaluation on Test Set
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Test Set Evaluation")
    print(f"{'='*60}")

    y_pred = best_model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"  MAE:  {mae:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.6f}")

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
        "Random Forest Regressor — Feature Importance",
        os.path.join(PLOTS_DIR, "regressor_feature_importance.png"),
    )

    # Actual vs Predicted
    plot_actual_vs_predicted(
        y_test, y_pred,
        f"Actual vs Predicted (R²={r2:.4f})",
        os.path.join(PLOTS_DIR, "regressor_actual_vs_predicted.png"),
    )

    # Residual distribution
    plot_residual_distribution(
        y_test, y_pred,
        "Regressor Residual Distribution",
        os.path.join(PLOTS_DIR, "regressor_residual_distribution.png"),
    )

    # ──────────────────────────────────────────────────────────────
    # Step 8: Save Model & Artifacts
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Saving Model & Artifacts")
    print(f"{'='*60}")

    joblib.dump(best_model, REGRESSOR_MODEL_PATH)
    print(f"  ✓ Model saved: {REGRESSOR_MODEL_PATH}")

    joblib.dump(scaler, SCALER_REG_PATH)
    print(f"  ✓ Scaler saved: {SCALER_REG_PATH}")

    save_feature_list(feature_cols, FEATURE_LIST_PATH)

    # ──────────────────────────────────────────────────────────────
    # Step 9: Save Metadata
    # ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    metadata = {
        "model_type": "RandomForestRegressor",
        "task": "regression",
        "target": REGRESSOR_TARGET,
        "features": feature_cols,
        "n_features": len(feature_cols),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "best_params": grid_search.best_params_,
        "metrics": {
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 6),
            "CV_R2_mean": round(cv_scores.mean(), 6),
            "CV_R2_std": round(cv_scores.std(), 6),
        },
        "feature_importances": {
            col: round(imp, 6)
            for col, imp in zip(feature_cols, best_model.feature_importances_)
        },
        "training_time_seconds": round(elapsed, 2),
        "validation_report": {
            "total_missing": validation_report["total_missing"],
            "duplicates": validation_report["duplicates"],
        },
    }
    save_metadata(metadata, METADATA_PATH)

    # ──────────────────────────────────────────────────────────────
    # Final Summary
    # ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("REGRESSOR TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Model:    {REGRESSOR_MODEL_PATH}")
    print(f"  R²:       {r2:.6f}")
    print(f"  RMSE:     {rmse:.4f}")
    print(f"  Time:     {elapsed:.1f}s")
    print()

    return best_model, scaler, metadata


if __name__ == "__main__":
    train_regressor()
