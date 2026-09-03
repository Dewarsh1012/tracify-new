"""
Inference module for the Blockchain Forensics ML Pipeline.
Provides reusable prediction functions for production use.

Usage:
    from inference import ForensicPredictor
    predictor = ForensicPredictor()
    result = predictor.predict_transaction_path(features)

    # Or as CLI:
    python src/inference.py
"""

import os
import sys
import json
import numpy as np
import joblib

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    REGRESSOR_MODEL_PATH, CLASSIFIER_MODEL_PATH,
    SCALER_REG_PATH, SCALER_CLF_PATH,
    FEATURE_COLUMNS, BEHAVIOR_LABELS,
    RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD,
)
from utils import get_risk_level, get_behavior_label


class ForensicPredictor:
    """
    Production-ready prediction interface for the Blockchain Forensics system.

    Loads trained Random Forest models and provides three prediction methods:
        1. predict_relevance()    — Relevance score + risk level
        2. predict_behavior()     — Behavior class + label
        3. predict_transaction_path() — Combined prediction (full report)
    """

    def __init__(self, models_dir: str = None):
        """
        Load trained models and scalers from disk.

        Args:
            models_dir: Path to models directory. Defaults to project/models/.
        """
        if models_dir:
            reg_path = os.path.join(models_dir, "random_forest_regressor.pkl")
            clf_path = os.path.join(models_dir, "random_forest_classifier.pkl")
            scaler_reg_path = os.path.join(models_dir, "scaler_regressor.pkl")
            scaler_clf_path = os.path.join(models_dir, "scaler_classifier.pkl")
        else:
            reg_path = REGRESSOR_MODEL_PATH
            clf_path = CLASSIFIER_MODEL_PATH
            scaler_reg_path = SCALER_REG_PATH
            scaler_clf_path = SCALER_CLF_PATH

        print("Loading models...")
        self.regressor = joblib.load(reg_path)
        self.classifier = joblib.load(clf_path)
        self.scaler_reg = joblib.load(scaler_reg_path)
        self.scaler_clf = joblib.load(scaler_clf_path)
        self.feature_names = FEATURE_COLUMNS
        print(f"  ✓ Regressor loaded from: {reg_path}")
        print(f"  ✓ Classifier loaded from: {clf_path}")
        print(f"  ✓ Models ready for inference")

    def _validate_features(self, features: dict) -> np.ndarray:
        """
        Validate and order feature dict into model-ready array.

        Args:
            features: Dict mapping feature names to values.

        Returns:
            Numpy array of shape (1, n_features).

        Raises:
            ValueError: If required features are missing.
        """
        missing = [f for f in self.feature_names if f not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        feature_vector = np.array(
            [[features[f] for f in self.feature_names]], dtype=np.float64
        )
        return feature_vector

    def predict_relevance(self, features: dict) -> dict:
        """
        Predict relevance score and risk level for a transaction path.

        Args:
            features: Dict with 12 feature values.

        Returns:
            {
                "relevance_score": float (0-100),
                "risk_level": str ("LOW" | "MEDIUM" | "HIGH")
            }
        """
        X = self._validate_features(features)
        X_scaled = self.scaler_reg.transform(X)
        score = float(self.regressor.predict(X_scaled)[0])
        score = max(0.0, min(100.0, round(score, 2)))

        return {
            "relevance_score": score,
            "risk_level": get_risk_level(score),
        }

    def predict_behavior(self, features: dict) -> dict:
        """
        Predict behavior class for a transaction path.

        Args:
            features: Dict with 12 feature values.

        Returns:
            {
                "behavior_class": int (0-3),
                "behavior_label": str
            }
        """
        X = self._validate_features(features)
        X_scaled = self.scaler_clf.transform(X)
        cls = int(self.classifier.predict(X_scaled)[0])

        return {
            "behavior_class": cls,
            "behavior_label": get_behavior_label(cls),
        }

    def predict_transaction_path(self, features: dict) -> dict:
        """
        Combined prediction: relevance + risk level + behavior classification.

        This is the primary API for the forensic investigation platform.

        Args:
            features: Dict with 12 feature values:
                - value_ratio, time_delta, same_asset, hop_count,
                  amount_similarity, degree, fanout, fanin,
                  address_age, transaction_frequency, entity_evidence,
                  path_length

        Returns:
            {
                "relevance_score": float,
                "risk_level": str,
                "behavior_class": int,
                "behavior_label": str
            }
        """
        relevance = self.predict_relevance(features)
        behavior = self.predict_behavior(features)

        return {
            "relevance_score": relevance["relevance_score"],
            "risk_level": relevance["risk_level"],
            "behavior_class": behavior["behavior_class"],
            "behavior_label": behavior["behavior_label"],
        }

    def predict_batch(self, features_list: list) -> list:
        """
        Batch prediction for multiple transaction paths.

        Args:
            features_list: List of feature dicts.

        Returns:
            List of prediction result dicts.
        """
        return [self.predict_transaction_path(f) for f in features_list]


# ══════════════════════════════════════════════════════════════════════
# CLI Demo
# ══════════════════════════════════════════════════════════════════════

def run_demo():
    """Demonstrate the inference pipeline with sample data."""
    print("\n" + "=" * 60)
    print("BLOCKCHAIN FORENSICS — INFERENCE DEMO")
    print("=" * 60)

    predictor = ForensicPredictor()

    # Sample transaction paths
    samples = [
        {
            "name": "High-risk fragmentation path",
            "features": {
                "value_ratio": 0.34, "time_delta": 15, "same_asset": 1,
                "hop_count": 2, "amount_similarity": 0.42, "degree": 10,
                "fanout": 6, "fanin": 1, "address_age": 220,
                "transaction_frequency": 120, "entity_evidence": 1,
                "path_length": 3,
            }
        },
        {
            "name": "Normal low-risk path",
            "features": {
                "value_ratio": 0.85, "time_delta": 3600, "same_asset": 1,
                "hop_count": 1, "amount_similarity": 0.91, "degree": 3,
                "fanout": 1, "fanin": 1, "address_age": 500,
                "transaction_frequency": 5, "entity_evidence": 0,
                "path_length": 2,
            }
        },
        {
            "name": "Rapid movement suspicious path",
            "features": {
                "value_ratio": 0.95, "time_delta": 120, "same_asset": 1,
                "hop_count": 3, "amount_similarity": 0.92, "degree": 8,
                "fanout": 2, "fanin": 1, "address_age": 400,
                "transaction_frequency": 22, "entity_evidence": 3,
                "path_length": 4,
            }
        },
        {
            "name": "Consolidation pattern",
            "features": {
                "value_ratio": 0.60, "time_delta": 7200, "same_asset": 1,
                "hop_count": 2, "amount_similarity": 0.75, "degree": 45,
                "fanout": 1, "fanin": 40, "address_age": 300,
                "transaction_frequency": 80, "entity_evidence": 2,
                "path_length": 3,
            }
        },
    ]

    for sample in samples:
        print(f"\n{'─'*50}")
        print(f"  Path: {sample['name']}")
        print(f"{'─'*50}")

        result = predictor.predict_transaction_path(sample["features"])

        print(f"  Relevance Score:  {result['relevance_score']}")
        print(f"  Risk Level:       {result['risk_level']}")
        print(f"  Behavior Class:   {result['behavior_class']}")
        print(f"  Behavior Label:   {result['behavior_label']}")

    print(f"\n{'='*60}")
    print("Demo complete ✓")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_demo()
