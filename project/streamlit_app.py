"""
Streamlit Web Application for Blockchain Forensics ML Platform - Simple Clean Theme.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as gg

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import FEATURE_COLUMNS, BEHAVIOR_LABELS, MODELS_DIR, PLOTS_DIR
from src.inference import ForensicPredictor

# Page Configuration
st.set_page_config(
    page_title="TRACIFY - Blockchain Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_forensic_predictor():
    """Load models once and cache across user sessions."""
    return ForensicPredictor(models_dir=MODELS_DIR)


@st.cache_data
def load_metadata():
    """Load metadata JSON reports."""
    meta = {}
    report_path = os.path.join(MODELS_DIR, "global_explanation_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            meta["explanation"] = json.load(f)
    return meta


predictor = load_forensic_predictor()
metadata = load_metadata()

# ──────────────────────────────────────────────────────────────────────
# Sidebar Controls & Scenario Presets
# ──────────────────────────────────────────────────────────────────────

st.sidebar.title("TRACIFY")
st.sidebar.caption("Blockchain Intelligence")

st.sidebar.markdown("### Investigator Presets")
preset_choice = st.sidebar.selectbox(
    "Load Scenario Preset:",
    [
        "Custom Input",
        "High-Risk Fragmentation",
        "Rapid Movement",
        "High Fanin Consolidation",
        "Normal Low-Risk Flow"
    ]
)

# Preset feature values
preset_values = {
    "value_ratio": 0.95,
    "time_delta": 120.0,
    "hop_count": 3,
    "amount_similarity": 0.92,
    "degree": 8.0,
    "fanout": 2.0,
    "fanin": 1.0,
    "transaction_frequency": 22.0,
    "entity_evidence": 3,
    "address_age": 400.0,
}

if preset_choice == "High-Risk Fragmentation":
    preset_values = {
        "value_ratio": 0.34, "time_delta": 15.0, "hop_count": 2,
        "amount_similarity": 0.42, "degree": 10.0, "fanout": 6.0,
        "fanin": 1.0, "transaction_frequency": 140.0,
        "entity_evidence": 2, "address_age": 220.0
    }
elif preset_choice == "Rapid Movement":
    preset_values = {
        "value_ratio": 0.95, "time_delta": 120.0, "hop_count": 4,
        "amount_similarity": 0.92, "degree": 8.0, "fanout": 2.0,
        "fanin": 1.0, "transaction_frequency": 120.0,
        "entity_evidence": 3, "address_age": 400.0
    }
elif preset_choice == "High Fanin Consolidation":
    preset_values = {
        "value_ratio": 0.60, "time_delta": 7200.0, "hop_count": 2,
        "amount_similarity": 0.75, "degree": 45.0, "fanout": 1.0,
        "fanin": 40.0, "transaction_frequency": 80.0,
        "entity_evidence": 2, "address_age": 300.0
    }
elif preset_choice == "Normal Low-Risk Flow":
    preset_values = {
        "value_ratio": 0.85, "time_delta": 3600.0, "hop_count": 1,
        "amount_similarity": 0.91, "degree": 3.0, "fanout": 1.0,
        "fanin": 1.0, "transaction_frequency": 5.0,
        "entity_evidence": 0, "address_age": 500.0
    }

st.sidebar.markdown("---")
st.sidebar.markdown("### Input Path Features")

value_ratio = st.sidebar.slider("value_ratio", 0.0, 1.0, float(preset_values["value_ratio"]), 0.01)
time_delta = st.sidebar.number_input("time_delta (sec)", 1.0, 50000.0, float(preset_values["time_delta"]), 10.0)
hop_count = st.sidebar.slider("hop_count", 1, 6, int(preset_values["hop_count"]))
amount_similarity = st.sidebar.slider("amount_similarity", 0.0, 1.0, float(preset_values["amount_similarity"]), 0.01)
degree = st.sidebar.number_input("degree (avg node degree)", 0.5, 300.0, float(preset_values["degree"]), 0.5)
fanout = st.sidebar.number_input("fanout (outgoing neighbors)", 0.0, 300.0, float(preset_values["fanout"]), 0.5)
fanin = st.sidebar.number_input("fanin (incoming neighbors)", 0.0, 300.0, float(preset_values["fanin"]), 0.5)
transaction_frequency = st.sidebar.number_input("transaction_frequency", 0.1, 2000.0, float(preset_values["transaction_frequency"]), 1.0)
entity_evidence = st.sidebar.selectbox("entity_evidence", [0, 1, 2, 3], index=int(preset_values["entity_evidence"]))
address_age = st.sidebar.number_input("address_age (days)", 1.0, 2000.0, float(preset_values["address_age"]), 10.0)

input_features = {
    "value_ratio": value_ratio,
    "time_delta": time_delta,
    "same_asset": 1,
    "hop_count": hop_count,
    "amount_similarity": amount_similarity,
    "degree": degree,
    "fanout": fanout,
    "fanin": fanin,
    "address_age": address_age,
    "transaction_frequency": transaction_frequency,
    "entity_evidence": entity_evidence,
    "path_length": hop_count + 1
}

# ──────────────────────────────────────────────────────────────────────
# Main Application Content
# ──────────────────────────────────────────────────────────────────────

st.title("TRACIFY - Blockchain Intelligence")
st.markdown("Forensic ML analysis and risk scoring for Bitcoin transaction paths.")

tab1, tab2, tab3 = st.tabs([
    "Path Investigation",
    "Model Performance & Plots",
    "Global Explainability"
])

# ── TAB 1: Single Path Investigation ──
with tab1:
    st.subheader("Transaction Path Risk Analysis")

    # Run Prediction
    prediction = predictor.predict_transaction_path(input_features)

    relevance_score = prediction["relevance_score"]
    risk_level = prediction["risk_level"]
    behavior_class = prediction["behavior_class"]
    behavior_label = prediction["behavior_label"]

    # Metrics Row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Relevance Score (0-100)",
            value=f"{relevance_score:.1f}",
            delta="High Priority Lead" if relevance_score >= 80 else "Standard Lead"
        )

    with col2:
        st.metric(
            label="Risk Level",
            value=f"{risk_level} ALERT"
        )

    with col3:
        st.metric(
            label="Behavioral Classification",
            value=behavior_label,
            delta=f"Class {behavior_class}"
        )

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("Network Graph Path Topology")
        
        # Build path node list
        node_labels = ["Suspect Origin"]
        for i in range(1, hop_count):
            node_labels.append(f"Wallet {chr(64 + i)}")
        node_labels.append("Target Destination")

        # Create Plotly graph
        x_coords = list(range(len(node_labels)))
        y_coords = [0] * len(node_labels)

        fig_net = gg.Figure()
        # Edge lines
        fig_net.add_trace(gg.Scatter(
            x=x_coords, y=y_coords, mode='lines',
            line=dict(color='#8b5cf6', width=3),
            hoverinfo='none'
        ))
        # Nodes
        fig_net.add_trace(gg.Scatter(
            x=x_coords, y=y_coords, mode='markers+text',
            marker=dict(size=36, color=['#ef4444'] + ['#8b5cf6']*(len(node_labels)-2) + ['#10b981']),
            text=node_labels, textposition="top center"
        ))
        fig_net.update_layout(
            showlegend=False,
            height=280,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        st.plotly_chart(fig_net, use_container_width=True)

    with col_right:
        st.subheader("Input Path Features Summary")
        summary_df = pd.DataFrame([
            {"Feature": k, "Value": v} for k, v in input_features.items()
        ])
        st.dataframe(summary_df, use_container_width=True, height=280)

# ── TAB 2: Model Performance & Plots ──
with tab2:
    st.subheader("Random Forest Models Performance Metrics")

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Random Forest Regressor**  
        * **Target**: `relevance_score` (0-100)  
        * **R² Score**: `0.9925` (99.25% Variance Explained)  
        * **RMSE**: `2.1231`  
        * **MAE**: `1.5861`  
        * **5-Fold CV Mean R²**: `0.9916 ± 0.0001`
        """)

    with col2:
        st.info("""
        **Random Forest Classifier**  
        * **Target**: `behavior_class` (0-3)  
        * **Accuracy**: `78.31%`  
        * **Multiclass ROC-AUC**: `85.65%`  
        * **Weighted Precision**: `82.76%`  
        * **5-Fold CV Mean F1**: `0.7178 ± 0.0014`
        """)

    st.markdown("---")
    st.subheader("Saved Evaluation Plots")

    plot_files = [
        ("regressor_actual_vs_predicted.png", "Regressor: Actual vs Predicted"),
        ("classifier_confusion_matrix.png", "Classifier: Confusion Matrix"),
        ("explainability_importance_comparison.png", "Feature Importance Comparison")
    ]

    cols = st.columns(len(plot_files))
    for i, (p_file, title) in enumerate(plot_files):
        p_path = os.path.join(PLOTS_DIR, p_file)
        if os.path.exists(p_path):
            with cols[i]:
                st.image(p_path, caption=title, use_container_width=True)

# ── TAB 3: Global Explainability ──
with tab3:
    st.subheader("Global Feature Importance & Insights")

    expl_data = metadata.get("explanation", {})

    col_reg, col_clf = st.columns(2)

    with col_reg:
        st.markdown("#### Top Features - Regressor (relevance_score)")
        reg_top = expl_data.get("regressor", {}).get("top_features", [])
        if reg_top:
            st.dataframe(pd.DataFrame(reg_top), use_container_width=True)

    with col_clf:
        st.markdown("#### Top Features - Classifier (behavior_class)")
        clf_top = expl_data.get("classifier", {}).get("top_features", [])
        if clf_top:
            st.dataframe(pd.DataFrame(clf_top), use_container_width=True)
