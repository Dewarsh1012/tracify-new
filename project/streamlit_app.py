"""
Streamlit Web Application for Blockchain Forensics ML Platform - Tracify Theme.
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
    page_title="TRACIFY - Blockchain Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────
# Custom CSS Styling (Tracify Dark Dashboard Aesthetic)
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b0d17 !important;
        color: #f8fafc;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Hide Streamlit Header / Footer Brand */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #080a14 !important;
        border-right: 1px solid #1a1e30;
    }

    /* Branding Header in Sidebar */
    .brand-header {
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid #1a1e30;
        margin-bottom: 1.5rem;
    }
    .brand-title {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        background: linear-gradient(135deg, #ffffff 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #64748b;
        margin-top: 0.2rem;
        font-weight: 600;
    }

    /* Top Command Center Header */
    .top-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #121527;
        border: 1px solid #1e2438;
        border-radius: 16px;
        padding: 0.8rem 1.4rem;
        margin-bottom: 1.2rem;
    }
    .search-mock {
        background-color: #0b0d17;
        border: 1px solid #232a40;
        border-radius: 20px;
        padding: 0.45rem 1rem;
        color: #64748b;
        font-size: 0.88rem;
        width: 380px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .user-badge {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .avatar-circle {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        color: #ffffff;
        box-shadow: 0 0 12px rgba(139, 92, 246, 0.4);
    }

    /* Command Center Hero Box */
    .command-hero {
        background: linear-gradient(135deg, #121527 0%, #161a32 100%);
        border: 1px solid #1e2438;
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
        position: relative;
    }
    .hero-tag {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #8b5cf6;
        margin-bottom: 0.4rem;
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.4rem;
    }
    .hero-user {
        color: #a855f7;
    }
    .hero-sub {
        font-size: 0.9rem;
        color: #94a3b8;
        margin: 0;
    }

    /* Sleek Cards */
    .tracify-card {
        background-color: #121527;
        border: 1px solid #1e2438;
        border-radius: 14px;
        padding: 1.2rem;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .tracify-card:hover {
        border-color: #3b4261;
    }
    .card-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.6rem;
    }
    .card-val {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
    }
    .card-change-positive {
        font-size: 0.8rem;
        color: #10b981;
        font-weight: 600;
        margin-top: 0.4rem;
    }
    .card-change-negative {
        font-size: 0.8rem;
        color: #ef4444;
        font-weight: 600;
        margin-top: 0.4rem;
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #080a14;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #1a1e30;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0 20px;
        background-color: transparent;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e2438 0%, #252c48 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }

    /* Risk Metrics Box */
    .risk-badge-high {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-medium {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-badge-low {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }

    /* Form & Input Adjustments */
    div[data-baseweb="select"] > div {
        background-color: #121527 !important;
        border-color: #232a40 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    .stSlider [data-baseweb="slider"] {
        margin-top: 0.5rem;
    }

    /* Progress bar style */
    .progress-bar-container {
        margin-bottom: 0.75rem;
    }
    .progress-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 0.25rem;
    }
    .progress-track {
        height: 6px;
        background-color: #1e2438;
        border-radius: 3px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 3px;
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
# Sidebar Controls & Scenario Presets (NO EMOJIS)
# ──────────────────────────────────────────────────────────────────────

st.sidebar.markdown("""
<div class="brand-header">
    <div class="brand-title">TRACIFY</div>
    <div class="brand-subtitle">Blockchain Intelligence</div>
</div>
""", unsafe_allow_html=True)

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

# Top Header Bar (Matching Image)
st.markdown("""
<div class="top-header-bar">
    <div class="search-mock">
        <span>Search cases, wallets, entities, tx hashes...</span>
        <span style="background: #1a1e30; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; color: #94a3b8;">Ctrl K</span>
    </div>
    <div class="user-badge">
        <div style="text-align: right;">
            <div style="font-weight: 700; font-size: 0.85rem; color: #ffffff;">BTAD24O1021 DEWARSH JAIN</div>
            <div style="font-size: 0.72rem; color: #64748b; text-transform: uppercase;">investigator</div>
        </div>
        <div class="avatar-circle">BT</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Command Center Hero Box (Matching Image)
st.markdown("""
<div class="command-hero">
    <div class="hero-tag">Investigator Command Center</div>
    <div class="hero-title">Good morning, <span class="hero-user">BTAD24O1021 DEWARSH JAIN</span></div>
    <p class="hero-sub">8 traces are building right now, and 12 findings are waiting on your review.</p>
</div>
""", unsafe_allow_html=True)

# 4 Key Dashboard Cards (Matching Image)
col_c1, col_c2, col_c3, col_c4 = st.columns(4)

with col_c1:
    st.markdown("""
    <div class="tracify-card">
        <div class="card-label">Active Investigations</div>
        <div class="card-val">08</div>
        <div class="card-change-positive">+12% from yesterday</div>
    </div>
    """, unsafe_allow_html=True)

with col_c2:
    st.markdown("""
    <div class="tracify-card">
        <div class="card-label">High Priority Cases</div>
        <div class="card-val">03</div>
        <div class="card-change-positive">+8% new escalations</div>
    </div>
    """, unsafe_allow_html=True)

with col_c3:
    st.markdown("""
    <div class="tracify-card">
        <div class="card-label">Findings to Review</div>
        <div class="card-val">12</div>
        <div class="card-change-negative">-4% vs last week</div>
    </div>
    """, unsafe_allow_html=True)

with col_c4:
    st.markdown("""
    <div class="tracify-card">
        <div class="card-label">Evidence Items</div>
        <div class="card-val">12</div>
        <div class="card-change-positive">+24% pinned this week</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Dashboard Charts Row (Investigation Overview + Workspace Risk Score)
chart_col1, chart_col2 = st.columns([1.6, 1])

with chart_col1:
    st.markdown("""
    <div style="background-color: #121527; border: 1px solid #1e2438; border-radius: 14px; padding: 1.2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div>
                <div style="font-weight: 700; font-size: 1rem; color: #ffffff;">Investigation overview</div>
                <div style="font-size: 0.8rem; color: #64748b;">Completed vs ongoing traces this week</div>
            </div>
            <div style="background: #1a1e30; color: #8b5cf6; padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">This week</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Plotly Spline Chart matching image
    days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    completed_vals = [1.0, 1.0, 2.0, 1.0, 0.2, 2.7, 2.2]
    ongoing_vals = [0.8, 0.8, 0.2, 1.0, 0.0, 2.5, 0.2]

    fig_overview = gg.Figure()
    fig_overview.add_trace(gg.Scatter(
        x=days, y=completed_vals, name="Completed",
        mode='lines', line=dict(color='#6366f1', width=3, shape='spline'),
        fill='tozeroy', fillcolor='rgba(99, 102, 241, 0.1)'
    ))
    fig_overview.add_trace(gg.Scatter(
        x=days, y=ongoing_vals, name="Ongoing",
        mode='lines', line=dict(color='#a855f7', width=3, shape='spline'),
        fill='tozeroy', fillcolor='rgba(168, 85, 247, 0.1)'
    ))

    fig_overview.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#94a3b8")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color='#64748b')),
        yaxis=dict(showgrid=True, gridcolor='#1e2438', zeroline=False, tickfont=dict(color='#64748b'))
    )
    st.plotly_chart(fig_overview, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with chart_col2:
    st.markdown("""
    <div style="background-color: #121527; border: 1px solid #1e2438; border-radius: 14px; padding: 1.2rem; height: 100%;">
        <div style="font-weight: 700; font-size: 1rem; color: #ffffff;">Workspace risk score</div>
        <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 1rem;">Weighted across all open investigations</div>
        <div style="display: flex; align-items: center; justify-content: space-around;">
            <div style="text-align: center;">
                <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff;">39</div>
                <div style="font-size: 0.75rem; color: #64748b;">Risk score</div>
                <div style="font-size: 0.75rem; color: #10b981; font-weight: 700;">Low risk</div>
            </div>
            <div style="width: 55%;">
                <div class="progress-bar-container">
                    <div class="progress-label"><span>Transaction behaviour</span><span>86</span></div>
                    <div class="progress-track"><div class="progress-fill" style="width: 86%; background: #ef4444;"></div></div>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-label"><span>Counterparty risk</span><span>21</span></div>
                    <div class="progress-track"><div class="progress-fill" style="width: 21%; background: #f59e0b;"></div></div>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-label"><span>Velocity</span><span>55</span></div>
                    <div class="progress-track"><div class="progress-fill" style="width: 55%; background: #8b5cf6;"></div></div>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-label"><span>Entity association</span><span>29</span></div>
                    <div class="progress-track"><div class="progress-fill" style="width: 29%; background: #6366f1;"></div></div>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-label"><span>Sanctions screening</span><span>5</span></div>
                    <div class="progress-track"><div class="progress-fill" style="width: 5%; background: #10b981;"></div></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# Main Feature Tabs (No Emojis, Batch Predictor CSV Removed)
# ──────────────────────────────────────────────────────────────────────

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

    # Top KPI Metrics Row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Relevance Score (0-100)",
            value=f"{relevance_score:.1f}",
            delta="High Priority Lead" if relevance_score >= 80 else "Standard Lead"
        )

    with col2:
        risk_class = "risk-badge-high" if risk_level == "HIGH" else "risk-badge-medium" if risk_level == "MEDIUM" else "risk-badge-low"
        st.markdown(f"""
        <div style="background-color: #121527; border: 1px solid #1e2438; border-radius: 12px; padding: 0.9rem; text-align: center;">
            <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">Risk Level</div>
            <div style="margin-top: 0.4rem;"><span class="{risk_class}">{risk_level} ALERT</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.metric(
            label="Behavioral Classification",
            value=behavior_label,
            delta=f"Class {behavior_class}"
        )

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

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
        st.subheader("Feature Feature Inputs Summary")
        summary_df = pd.DataFrame([
            {"Feature": k, "Value": v} for k, v in input_features.items()
        ])
        st.dataframe(summary_df, use_container_width=True, height=260)

# ── TAB 2: Model Performance & Plots ──
with tab2:
    st.subheader("Random Forest Models Performance Metrics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background-color: #121527; border: 1px solid #1e2438; border-radius: 12px; padding: 1.2rem;">
            <h4 style="color: #a855f7; margin-top: 0;">Random Forest Regressor</h4>
            <p><strong>Target</strong>: <code>relevance_score</code> (0-100)</p>
            <p><strong>R2 Score</strong>: <code>0.9925</code> (99.25% Variance Explained)</p>
            <p><strong>RMSE</strong>: <code>2.1231</code></p>
            <p><strong>MAE</strong>: <code>1.5861</code></p>
            <p><strong>5-Fold CV Mean R2</strong>: <code>0.9916 +- 0.0001</code></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background-color: #121527; border: 1px solid #1e2438; border-radius: 12px; padding: 1.2rem;">
            <h4 style="color: #6366f1; margin-top: 0;">Random Forest Classifier</h4>
            <p><strong>Target</strong>: <code>behavior_class</code> (0-3)</p>
            <p><strong>Accuracy</strong>: <code>78.31%</code></p>
            <p><strong>Multiclass ROC-AUC</strong>: <code>85.65%</code></p>
            <p><strong>Weighted Precision</strong>: <code>82.76%</code></p>
            <p><strong>5-Fold CV Mean F1</strong>: <code>0.7178 +- 0.0014</code></p>
        </div>
        """, unsafe_allow_html=True)

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
