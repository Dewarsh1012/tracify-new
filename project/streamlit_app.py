"""
Streamlit Web Application for Blockchain Forensics ML Platform.
Suitable for 1-click deployment on Streamlit Community Cloud (100% Free).

Usage:
    streamlit run streamlit_app.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as gg

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import FEATURE_COLUMNS, BEHAVIOR_LABELS, MODELS_DIR, PLOTS_DIR
from src.inference import ForensicPredictor

# Page Configuration
st.set_page_config(
    page_title="Blockchain Forensics ML Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-box {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .high-risk {
        color: #f43f5e !important;
        font-weight: bold;
    }
    .medium-risk {
        color: #f59e0b !important;
        font-weight: bold;
    }
    .low-risk {
        color: #10b981 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


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

st.sidebar.title("🛡️ Blockchain Forensics")
st.sidebar.markdown("**NetworkX Transaction Graph ML Investigator**")

st.sidebar.subheader("🎯 Investigator Presets")
preset_choice = st.sidebar.selectbox(
    "Load Scenario Preset:",
    [
        "Custom Input",
        "🔴 High-Risk Fragmentation",
        "⚡ Rapid Movement",
        "🧩 High Fanin Consolidation",
        "🟢 Normal Low-Risk Flow"
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

if preset_choice == "🔴 High-Risk Fragmentation":
    preset_values = {
        "value_ratio": 0.34, "time_delta": 15.0, "hop_count": 2,
        "amount_similarity": 0.42, "degree": 10.0, "fanout": 6.0,
        "fanin": 1.0, "transaction_frequency": 140.0,
        "entity_evidence": 2, "address_age": 220.0
    }
elif preset_choice == "⚡ Rapid Movement":
    preset_values = {
        "value_ratio": 0.95, "time_delta": 120.0, "hop_count": 4,
        "amount_similarity": 0.92, "degree": 8.0, "fanout": 2.0,
        "fanin": 1.0, "transaction_frequency": 120.0,
        "entity_evidence": 3, "address_age": 400.0
    }
elif preset_choice == "🧩 High Fanin Consolidation":
    preset_values = {
        "value_ratio": 0.60, "time_delta": 7200.0, "hop_count": 2,
        "amount_similarity": 0.75, "degree": 45.0, "fanout": 1.0,
        "fanin": 40.0, "transaction_frequency": 80.0,
        "entity_evidence": 2, "address_age": 300.0
    }
elif preset_choice == "🟢 Normal Low-Risk Flow":
    preset_values = {
        "value_ratio": 0.85, "time_delta": 3600.0, "hop_count": 1,
        "amount_similarity": 0.91, "degree": 3.0, "fanout": 1.0,
        "fanin": 1.0, "transaction_frequency": 5.0,
        "entity_evidence": 0, "address_age": 500.0
    }

st.sidebar.subheader("🎛️ Input Path Features")

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

st.title("🛡️ Blockchain Forensics ML Platform")
st.markdown("Automated **Relevance Ranking** and **Behavioral Classification** for Suspect Wallet Paths")

# Create Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Path Investigation",
    "📊 Model Performance & Plots",
    "📜 Global Explainability",
    "📁 Batch Predictor (CSV)"
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

    # Top KPI Metrics Row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Relevance Score (0–100)",
            value=f"{relevance_score:.1f}",
            delta="High Priority Lead" if relevance_score >= 80 else "Standard Lead"
        )

    with col2:
        risk_color = "red" if risk_level == "HIGH" else "orange" if risk_level == "MEDIUM" else "green"
        st.metric(
            label="Risk Level",
            value=risk_level,
            delta=f"{risk_color.upper()} ALERT"
        )

    with col3:
        st.metric(
            label="Behavioral Classification",
            value=behavior_label,
            delta=f"Class {behavior_class}"
        )

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🌐 Network Graph Path Topology")
        
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
            line=dict(color='#06b6d4', width=3),
            hoverinfo='none'
        ))
        # Nodes
        fig_net.add_trace(gg.Scatter(
            x=x_coords, y=y_coords, mode='markers+text',
            marker=dict(size=36, color=['#f43f5e'] + ['#8b5cf6']*(len(node_labels)-2) + ['#10b981']),
            text=node_labels, textposition="top center",
            textfont=dict(size=12, color="white")
        ))
        fig_net.update_layout(
            showlegend=False,
            height=260,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_net, use_container_width=True)

    with col_right:
        st.subheader("🎯 Feature Contribution Breakdown")
        
        reg_imp = metadata.get("explanation", {}).get("regressor", {}).get("top_features", [])
        
        imp_df = pd.DataFrame(reg_imp)
        if not imp_df.empty:
            fig_bar = px.bar(
                imp_df.sort_values("importance", ascending=True),
                x="importance", y="feature",
                orientation='h',
                title="Global Regressor Feature Importance",
                color_discrete_sequence=['#06b6d4']
            )
            fig_bar.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

# ── TAB 2: Model Performance & Plots ──
with tab2:
    st.subheader("Random Forest Models Performance Metrics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Random Forest Regressor")
        st.write("- **Target**: `relevance_score` (0–100)")
        st.write("- **R² Score**: `0.9925` (99.25% Variance Explained)")
        st.write("- **RMSE**: `2.1231`")
        st.write("- **MAE**: `1.5861`")
        st.write("- **5-Fold CV Mean R²**: `0.9916 ± 0.0001`")

    with col2:
        st.markdown("### 🎯 Random Forest Classifier")
        st.write("- **Target**: `behavior_class` (0–3)")
        st.write("- **Accuracy**: `78.31%`")
        st.write("- **Multiclass ROC-AUC**: `85.65%`")
        st.write("- **Weighted Precision**: `82.76%`")
        st.write("- **5-Fold CV Mean F1**: `0.7178 ± 0.0014`")

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
        st.markdown("#### Top Features — Regressor (`relevance_score`)")
        reg_top = expl_data.get("regressor", {}).get("top_features", [])
        if reg_top:
            st.dataframe(pd.DataFrame(reg_top), use_container_width=True)

    with col_clf:
        st.markdown("#### Top Features — Classifier (`behavior_class`)")
        clf_top = expl_data.get("classifier", {}).get("top_features", [])
        if clf_top:
            st.dataframe(pd.DataFrame(clf_top), use_container_width=True)

# ── TAB 4: Batch Predictor ──
with tab4:
    st.subheader("📁 Batch Path Investigation (Upload CSV)")
    st.markdown("Upload a CSV file containing the 12 feature columns to generate predictions for multiple paths at once.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_df):,}** records.")

            missing_cols = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}")
            else:
                if st.button("Run Batch Prediction"):
                    records = batch_df[FEATURE_COLUMNS].to_dict(orient="records")
                    results = predictor.predict_batch(records)

                    res_df = pd.DataFrame(results)
                    final_df = pd.concat([batch_df, res_df], axis=1)

                    st.success("Batch Prediction Complete!")
                    st.dataframe(final_df.head(20), use_container_width=True)

                    # Download button
                    csv_data = final_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_data,
                        file_name="forensic_batch_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")
