"""
Flask Web Application & REST API Server for Blockchain Forensics Platform.
Provides an interactive localhost UI for analyzing transaction paths.

Usage:
    python src/app.py
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FEATURE_COLUMNS, BEHAVIOR_LABELS, PROJECT_ROOT, MODELS_DIR
from inference import ForensicPredictor

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static"),
)

# Initialize predictor globally
predictor = None
metadata_cache = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = ForensicPredictor(models_dir=MODELS_DIR)
    return predictor


def get_metadata():
    global metadata_cache
    if metadata_cache is None:
        report_path = os.path.join(MODELS_DIR, "global_explanation_report.json")
        reg_meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
        clf_meta_path = os.path.join(MODELS_DIR, "classifier_metadata.json")

        metadata_cache = {
            "explanation": {},
            "regressor": {},
            "classifier": {},
        }

        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                metadata_cache["explanation"] = json.load(f)

        if os.path.exists(reg_meta_path):
            with open(reg_meta_path, "r") as f:
                metadata_cache["regressor"] = json.load(f)

        if os.path.exists(clf_meta_path):
            with open(clf_meta_path, "r") as f:
                metadata_cache["classifier"] = json.load(f)

    return metadata_cache


# ──────────────────────────────────────────────────────────────────────
# Routes & API Endpoints
# ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the main forensics web dashboard UI."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Run predictions for a transaction path.

    Expects JSON body with 12 features:
        value_ratio, time_delta, same_asset, hop_count,
        amount_similarity, degree, fanout, fanin,
        address_age, transaction_frequency, entity_evidence, path_length
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input JSON provided"}), 400

        # Extract features
        features = {}
        for col in FEATURE_COLUMNS:
            if col not in data:
                return jsonify({"error": f"Missing feature: '{col}'"}), 400
            features[col] = float(data[col])

        # Run prediction
        pred_engine = get_predictor()
        result = pred_engine.predict_transaction_path(features)

        # Get top feature importances for explainability
        meta = get_metadata()
        reg_imp = meta.get("explanation", {}).get("regressor", {}).get("top_features", [])
        clf_imp = meta.get("explanation", {}).get("classifier", {}).get("top_features", [])

        # Add feature contribution estimates for this specific path
        contributions = []
        for col in FEATURE_COLUMNS:
            val = features[col]
            # Simple contribution metric: scaled value * regressor feature importance
            reg_feat_imp = next((item["importance"] for item in reg_imp if item["feature"] == col), 0.05)
            contributions.append({
                "feature": col,
                "value": val,
                "importance": reg_feat_imp,
                "contribution": round(val * reg_feat_imp, 4)
            })

        contributions = sorted(contributions, key=lambda x: x["importance"], reverse=True)

        response = {
            "status": "success",
            "prediction": result,
            "features_input": features,
            "feature_contributions": contributions,
            "top_regressor_features": reg_imp[:5],
            "top_classifier_features": clf_imp[:5],
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/presets", methods=["GET"])
def get_presets():
    """Return pre-configured investigation scenarios."""
    presets = {
        "rapid_movement": {
            "name": "⚡ Rapid Movement / High Velocity",
            "description": "Short time delta, high transaction frequency, multi-hop laundering pattern.",
            "features": {
                "value_ratio": 0.95,
                "time_delta": 120.0,
                "same_asset": 1,
                "hop_count": 4,
                "amount_similarity": 0.92,
                "degree": 8.0,
                "fanout": 2.0,
                "fanin": 1.0,
                "address_age": 400,
                "transaction_frequency": 120.0,
                "entity_evidence": 3,
                "path_length": 5
            }
        },
        "fragmentation": {
            "name": "🔴 High-Risk Fragmentation (Splitting)",
            "description": "High fanout node, splitting illicit funds across multiple recipient wallets.",
            "features": {
                "value_ratio": 0.34,
                "time_delta": 15.0,
                "same_asset": 1,
                "hop_count": 2,
                "amount_similarity": 0.42,
                "degree": 10.0,
                "fanout": 6.0,
                "fanin": 1.0,
                "address_age": 220,
                "transaction_frequency": 140.0,
                "entity_evidence": 2,
                "path_length": 3
            }
        },
        "consolidation": {
            "name": "🧩 High Fanin Consolidation",
            "description": "High fanin hub merging scattered funds into a central destination exchange/wallet.",
            "features": {
                "value_ratio": 0.60,
                "time_delta": 7200.0,
                "same_asset": 1,
                "hop_count": 2,
                "amount_similarity": 0.75,
                "degree": 45.0,
                "fanout": 1.0,
                "fanin": 40.0,
                "address_age": 300,
                "transaction_frequency": 80.0,
                "entity_evidence": 2,
                "path_length": 3
            }
        },
        "normal_flow": {
            "name": "🟢 Normal Low-Risk Flow",
            "description": "Standard peer-to-peer transaction with high stability and low node fanout.",
            "features": {
                "value_ratio": 0.85,
                "time_delta": 3600.0,
                "same_asset": 1,
                "hop_count": 1,
                "amount_similarity": 0.91,
                "degree": 3.0,
                "fanout": 1.0,
                "fanin": 1.0,
                "address_age": 500,
                "transaction_frequency": 5.0,
                "entity_evidence": 0,
                "path_length": 2
            }
        }
    }
    return jsonify(presets)


@app.route("/api/metadata", methods=["GET"])
def metadata():
    """Return model metrics and metadata."""
    meta = get_metadata()
    return jsonify({
        "status": "success",
        "metadata": meta,
        "features": FEATURE_COLUMNS,
        "behavior_labels": BEHAVIOR_LABELS
    })


if __name__ == "__main__":
    # Warm up models
    get_predictor()
    print("\n" + "═"*60)
    print("🚀 BLOCKCHAIN FORENSICS INVESTIGATION SYSTEM RUNNING ON LOCALHOST")
    print("   Open browser: http://127.0.0.1:5000")
    print("═"*60 + "\n")

    app.run(host="127.0.0.1", port=5000, debug=True)
