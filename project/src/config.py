"""
Configuration module for the Blockchain Forensics ML Pipeline.
Centralizes all paths, hyperparameters, and constants.
"""

import os

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

REGRESSOR_DATA = os.path.join(DATA_DIR, "forensic_regressor_dataset.csv")
CLASSIFIER_DATA = os.path.join(DATA_DIR, "forensic_classifier_dataset.csv")

REGRESSOR_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_regressor.pkl")
CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_classifier.pkl")
SCALER_REG_PATH = os.path.join(MODELS_DIR, "scaler_regressor.pkl")
SCALER_CLF_PATH = os.path.join(MODELS_DIR, "scaler_classifier.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_list.json")

# ──────────────────────────────────────────────────────────────────────
# Feature Schema
# ──────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "value_ratio",
    "time_delta",
    "same_asset",
    "hop_count",
    "amount_similarity",
    "degree",
    "fanout",
    "fanin",
    "address_age",
    "transaction_frequency",
    "entity_evidence",
    "path_length",
]

REGRESSOR_TARGET = "relevance_score"
CLASSIFIER_TARGET = "behavior_class"

# ──────────────────────────────────────────────────────────────────────
# Behavior Class Labels
# ──────────────────────────────────────────────────────────────────────
BEHAVIOR_LABELS = {
    0: "Normal Flow",
    1: "Rapid Movement",
    2: "Fragmentation",
    3: "Consolidation",
}

# ──────────────────────────────────────────────────────────────────────
# Risk Level Thresholds
# ──────────────────────────────────────────────────────────────────────
RISK_HIGH_THRESHOLD = 80
RISK_MEDIUM_THRESHOLD = 50

# ──────────────────────────────────────────────────────────────────────
# Train/Test Split
# ──────────────────────────────────────────────────────────────────────
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────────
# XGBoost GridSearchCV Hyperparameter Grids (Tuned against Overfitting/Underfitting)
# ──────────────────────────────────────────────────────────────────────
XGB_REG_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_alpha": [0.1],
    "reg_lambda": [1.0],
}

XGB_CLF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_alpha": [0.1],
    "reg_lambda": [1.0],
}

CV_FOLDS = 5

