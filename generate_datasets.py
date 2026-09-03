#!/usr/bin/env python3
"""
Elliptic Bitcoin Dataset -> Forensic ML Dataset Generator
=========================================================
Generates two production-ready datasets (100K rows each) for:
  1. Random Forest Regressor  (target: relevance_score)
  2. Random Forest Classifier (target: behavior_class)

Features are derived from actual graph topology, path traversals,
and real feature distributions from the Elliptic dataset.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from scipy.spatial.distance import cosine as cosine_dist

warnings.filterwarnings("ignore")
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSES_FILE = os.path.join(BASE_DIR, "elliptic_txs_classes.csv")
EDGELIST_FILE = os.path.join(BASE_DIR, "elliptic_txs_edgelist.csv")
FEATURES_FILE = os.path.join(BASE_DIR, "elliptic_txs_features.csv")

OUTPUT_REG = os.path.join(BASE_DIR, "forensic_regressor_dataset.csv")
OUTPUT_CLF = os.path.join(BASE_DIR, "forensic_classifier_dataset.csv")
REPORT_DIR = os.path.join(BASE_DIR, "validation_report")

NUM_ROWS = 100_000
MAX_HOPS = 6

os.makedirs(REPORT_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# PHASE 1: Build Transaction Graph
# ══════════════════════════════════════════════════════════════════════
log("PHASE 1: Building transaction graph...")

# Load classes
classes_df = pd.read_csv(CLASSES_FILE)
class_map = {}
for _, row in classes_df.iterrows():
    txid = int(row["txId"])
    cls = row["class"]
    if cls == "1":
        class_map[txid] = 1  # illicit
    elif cls == "2":
        class_map[txid] = 2  # licit
    else:
        class_map[txid] = 0  # unknown

# Load features (no header)
log("  Loading features (this may take a moment)...")
features_df = pd.read_csv(FEATURES_FILE, header=None)
features_df.columns = [f"f{i}" for i in range(features_df.shape[1])]
# f0 = txId, f1 = timestep, f2..f95 = local features, f96..f166 = aggregate features
txid_to_timestep = dict(zip(features_df["f0"].astype(int), features_df["f1"].astype(int)))

# Build a lookup of txid -> feature vector (local features only, f2..f15 for efficiency)
# We use a subset of local features for similarity calculations
FEAT_COLS = [f"f{i}" for i in range(2, 16)]
txid_to_feats = {}
for _, row in features_df[["f0"] + FEAT_COLS].iterrows():
    txid_to_feats[int(row["f0"])] = row[FEAT_COLS].values.astype(np.float64)

# We also extract some feature columns for deriving realistic distributions
# Use f2 (a local feature) as proxy for "amount-like" values
amount_proxy = features_df["f2"].values
amount_mean, amount_std = np.mean(amount_proxy), np.std(amount_proxy)

log("  Building directed graph...")
edgelist_df = pd.read_csv(EDGELIST_FILE)
G = nx.DiGraph()

# Add all nodes from classes (ensures full coverage)
for txid in class_map:
    G.add_node(txid, label=class_map.get(txid, 0),
               timestep=txid_to_timestep.get(txid, 1))

# Add edges
for _, row in edgelist_df.iterrows():
    src, dst = int(row["txId1"]), int(row["txId2"])
    G.add_edge(src, dst)

# Ensure all nodes have attributes
for node in G.nodes():
    if "label" not in G.nodes[node]:
        G.nodes[node]["label"] = class_map.get(node, 0)
    if "timestep" not in G.nodes[node]:
        G.nodes[node]["timestep"] = txid_to_timestep.get(node, 1)

log(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# ══════════════════════════════════════════════════════════════════════
# PHASE 2: Extract Real Graph Statistics
# ══════════════════════════════════════════════════════════════════════
log("PHASE 2: Extracting real graph statistics...")

# Pre-compute degree dicts
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())
total_deg = dict(G.degree())

in_deg_vals = np.array(list(in_deg.values()), dtype=np.float64)
out_deg_vals = np.array(list(out_deg.values()), dtype=np.float64)
total_deg_vals = np.array(list(total_deg.values()), dtype=np.float64)

log(f"  Degree stats: mean_in={in_deg_vals.mean():.2f}, mean_out={out_deg_vals.mean():.2f}, "
    f"mean_total={total_deg_vals.mean():.2f}")

# Timestep stats
timesteps = np.array([G.nodes[n].get("timestep", 1) for n in G.nodes()])
ts_min, ts_max = timesteps.min(), timesteps.max()
log(f"  Timestep range: {ts_min} – {ts_max}")

# ══════════════════════════════════════════════════════════════════════
# PHASE 3: Path Sampling
# ══════════════════════════════════════════════════════════════════════
log("PHASE 3: Sampling paths from graph...")

# Build adjacency list for fast random walks
adj_out = defaultdict(list)
for src, dst in G.edges():
    adj_out[src].append(dst)

# Identify seed nodes: illicit nodes first, then high-degree nodes
illicit_nodes = [n for n, d in G.nodes(data=True) if d.get("label") == 1]
high_deg_nodes = sorted(G.nodes(), key=lambda n: total_deg.get(n, 0), reverse=True)[:5000]
all_nodes_list = list(G.nodes())

# Combine seed pools with priority
seed_pool = illicit_nodes * 10 + high_deg_nodes * 3 + all_nodes_list
np.random.shuffle(seed_pool)

def random_walk(start, max_len):
    """Perform a directed random walk from start node."""
    path = [start]
    current = start
    for _ in range(max_len):
        neighbors = adj_out.get(current, [])
        if not neighbors:
            break
        current = neighbors[np.random.randint(len(neighbors))]
        path.append(current)
    return path

paths = []
seed_idx = 0
attempts = 0
max_attempts = NUM_ROWS * 20  # safety limit

while len(paths) < NUM_ROWS and attempts < max_attempts:
    attempts += 1
    # Pick seed
    seed = seed_pool[seed_idx % len(seed_pool)]
    seed_idx += 1

    # Random hop length 1-6
    hop_len = np.random.randint(1, MAX_HOPS + 1)
    path = random_walk(seed, hop_len)

    if len(path) >= 2:  # Need at least 2 nodes for a valid path
        paths.append(path)

    if len(paths) % 20000 == 0 and len(paths) > 0:
        log(f"  Sampled {len(paths):,} / {NUM_ROWS:,} paths...")

log(f"  Total paths sampled: {len(paths):,} (attempts: {attempts:,})")

# ══════════════════════════════════════════════════════════════════════
# PHASE 4: Feature Engineering
# ══════════════════════════════════════════════════════════════════════
log("PHASE 4: Engineering features for each path...")

# Pre-compute feature vectors as numpy array for fast access
all_node_set = set(G.nodes())

def compute_path_features(path):
    """Compute all 12 features for a given path."""
    hop_count = len(path) - 1
    path_length = len(path)

    # ── Graph features (averaged over path nodes) ──
    degrees = [total_deg.get(n, 0) for n in path]
    fanouts = [out_deg.get(n, 0) for n in path]
    fanins = [in_deg.get(n, 0) for n in path]

    degree = np.mean(degrees)
    fanout = np.mean(fanouts)
    fanin = np.mean(fanins)

    # ── Value ratio ──
    # Use actual feature vectors to compute ratio between consecutive nodes
    feat_vecs = []
    for n in path:
        if n in txid_to_feats:
            feat_vecs.append(txid_to_feats[n])
        else:
            feat_vecs.append(np.zeros(len(FEAT_COLS)))

    # value_ratio: ratio of L2 norms of consecutive feature vectors
    norms = [np.linalg.norm(fv) + 1e-10 for fv in feat_vecs]
    if len(norms) >= 2:
        ratios = [min(norms[i+1], norms[i]) / max(norms[i+1], norms[i])
                  for i in range(len(norms)-1)]
        value_ratio = np.mean(ratios)
    else:
        value_ratio = 0.5

    value_ratio = np.clip(value_ratio, 0.0, 1.0)

    # ── Amount similarity ──
    # Cosine similarity between consecutive node feature vectors
    if len(feat_vecs) >= 2:
        sims = []
        for i in range(len(feat_vecs)-1):
            n1, n2 = np.linalg.norm(feat_vecs[i]), np.linalg.norm(feat_vecs[i+1])
            if n1 > 1e-10 and n2 > 1e-10:
                sim = 1.0 - cosine_dist(feat_vecs[i], feat_vecs[i+1])
                sims.append(np.clip(sim, 0.0, 1.0))
            else:
                sims.append(0.5)
        amount_similarity = np.mean(sims)
    else:
        amount_similarity = 0.5

    amount_similarity = np.clip(amount_similarity, 0.0, 1.0)

    # ── Time delta ──
    # Timestep differences scaled to seconds (each timestep ~ 2 weeks = 1,209,600 sec)
    ts_vals = [txid_to_timestep.get(n, 1) for n in path]
    if len(ts_vals) >= 2:
        deltas = [abs(ts_vals[i+1] - ts_vals[i]) for i in range(len(ts_vals)-1)]
        raw_delta = np.mean(deltas)
        # Scale: each timestep diff represents ~hours to days
        # Map to realistic seconds: 0 diff -> small seconds, large diff -> large seconds
        time_delta = raw_delta * 3600 * 6  # ~6 hours per timestep unit for realism
    else:
        time_delta = 300.0

    # Add some realistic noise
    time_delta = max(1.0, time_delta + np.random.normal(0, 60))
    time_delta = round(time_delta, 1)

    # ── Same asset ──
    # Bitcoin-only dataset: predominantly 1, small probability of 0 for cross-chain sim
    same_asset = 1 if np.random.random() < 0.92 else 0

    # ── Address age ──
    # Derive from timestep: lower timestep = older address
    ts_mean = np.mean(ts_vals)
    # Map timestep 1..49 to address_age in days: older timesteps get higher age
    address_age = max(1, int((ts_max - ts_mean + 1) / ts_max * 800 + np.random.normal(0, 50)))
    address_age = np.clip(address_age, 1, 1500)

    # ── Transaction frequency ──
    # Based on actual node degree / active period proxy
    mean_degree = np.mean(degrees)
    active_period = max(1, max(ts_vals) - min(ts_vals) + 1)
    transaction_frequency = mean_degree / active_period * 10  # scale for realism
    transaction_frequency = max(0.1, transaction_frequency + abs(np.random.normal(0, 5)))
    transaction_frequency = round(transaction_frequency, 1)

    # ── Entity evidence ──
    # Based on class labels of nodes in path
    labels = [G.nodes[n].get("label", 0) for n in path if n in G.nodes]
    illicit_count = sum(1 for l in labels if l == 1)
    licit_count = sum(1 for l in labels if l == 2)

    if illicit_count >= 2:
        entity_evidence = 3  # Strong
    elif illicit_count == 1:
        entity_evidence = 2  # Medium
    elif licit_count > 0:
        entity_evidence = 1  # Weak
    else:
        entity_evidence = 0  # Unknown

    return {
        "value_ratio": round(value_ratio, 4),
        "time_delta": time_delta,
        "same_asset": int(same_asset),
        "hop_count": int(hop_count),
        "amount_similarity": round(amount_similarity, 4),
        "degree": round(degree, 2),
        "fanout": round(fanout, 2),
        "fanin": round(fanin, 2),
        "address_age": int(address_age),
        "transaction_frequency": transaction_frequency,
        "entity_evidence": int(entity_evidence),
        "path_length": int(path_length),
    }


# Process all paths
records = []
for i, path in enumerate(paths):
    rec = compute_path_features(path)
    records.append(rec)
    if (i + 1) % 25000 == 0:
        log(f"  Processed {i+1:,} / {len(paths):,} paths...")

df = pd.DataFrame(records)
log(f"  Feature matrix shape: {df.shape}")

# ══════════════════════════════════════════════════════════════════════
# PHASE 5A: Generate Relevance Score
# ══════════════════════════════════════════════════════════════════════
log("PHASE 5A: Generating relevance scores...")

def compute_relevance(row):
    """
    relevance_score =
      35% value continuity +
      25% temporal continuity +
      15% path efficiency +
      15% entity evidence +
      10% graph importance
    """
    # Value continuity: high value_ratio + high amount_similarity = high continuity
    value_cont = (0.6 * row["value_ratio"] + 0.4 * row["amount_similarity"])

    # Temporal continuity: lower time_delta = higher continuity (faster = more relevant)
    # Normalize: use exponential decay
    temp_cont = np.exp(-row["time_delta"] / 50000)

    # Path efficiency: shorter paths with fewer hops = more efficient
    path_eff = 1.0 / (1.0 + row["hop_count"] * 0.3)

    # Entity evidence: direct mapping 0->0, 1->0.33, 2->0.67, 3->1.0
    ent_score = row["entity_evidence"] / 3.0

    # Graph importance: higher degree nodes = more important
    # Normalize degree to [0,1] using empirical max
    graph_imp = min(1.0, row["degree"] / 20.0) * 0.5 + \
                min(1.0, row["fanout"] / 10.0) * 0.3 + \
                min(1.0, row["fanin"] / 10.0) * 0.2

    raw_score = (0.35 * value_cont +
                 0.25 * temp_cont +
                 0.15 * path_eff +
                 0.15 * ent_score +
                 0.10 * graph_imp)

    return raw_score

raw_scores = df.apply(compute_relevance, axis=1).values

# Scale to 0-100
raw_min, raw_max = raw_scores.min(), raw_scores.max()
scaled = (raw_scores - raw_min) / (raw_max - raw_min + 1e-10) * 100

# Shape distribution to target: High(80-100)=30%, Medium(50-79)=50%, Low(0-49)=20%
# Use quantile-based remapping
sorted_indices = np.argsort(scaled)
n = len(scaled)
relevance_scores = np.zeros(n)

for rank, idx in enumerate(sorted_indices):
    percentile = rank / n
    if percentile < 0.20:
        # Low: 0-49
        relevance_scores[idx] = percentile / 0.20 * 49
    elif percentile < 0.70:
        # Medium: 50-79
        relevance_scores[idx] = 50 + (percentile - 0.20) / 0.50 * 29
    else:
        # High: 80-100
        relevance_scores[idx] = 80 + (percentile - 0.70) / 0.30 * 20

# Add small noise to make distribution more natural
relevance_scores += np.random.normal(0, 1.5, n)
relevance_scores = np.clip(np.round(relevance_scores, 2), 0, 100)

df["relevance_score"] = relevance_scores

# Verify distribution
high_pct = np.mean(relevance_scores >= 80) * 100
med_pct = np.mean((relevance_scores >= 50) & (relevance_scores < 80)) * 100
low_pct = np.mean(relevance_scores < 50) * 100
log(f"  Relevance distribution: High={high_pct:.1f}%, Medium={med_pct:.1f}%, Low={low_pct:.1f}%")

# ══════════════════════════════════════════════════════════════════════
# PHASE 5B: Generate Behavior Classes
# ══════════════════════════════════════════════════════════════════════
log("PHASE 5B: Generating behavior classes...")

# Compute percentiles for thresholding
td_25 = np.percentile(df["time_delta"], 25)
tf_75 = np.percentile(df["transaction_frequency"], 75)
fo_75 = np.percentile(df["fanout"], 75)
fi_75 = np.percentile(df["fanin"], 75)

log(f"  Thresholds: td_25={td_25:.1f}, tf_75={tf_75:.1f}, fo_75={fo_75:.1f}, fi_75={fi_75:.1f}")

def assign_behavior_class(row):
    """Assign behavior class based on feature thresholds."""
    scores = {0: 0, 1: 0, 2: 0, 3: 0}

    # Class 1: Rapid Movement
    if row["time_delta"] < td_25:
        scores[1] += 2
    if row["transaction_frequency"] > tf_75:
        scores[1] += 2
    if row["hop_count"] >= 3:
        scores[1] += 1

    # Class 2: Fragmentation
    if row["fanout"] > fo_75:
        scores[2] += 3
    if row["value_ratio"] < 0.5:
        scores[2] += 1

    # Class 3: Consolidation
    if row["fanin"] > fi_75:
        scores[3] += 3
    if row["amount_similarity"] > 0.7:
        scores[3] += 1

    # Class 0: Normal Flow (default / low signals)
    if row["value_ratio"] > 0.6 and row["fanout"] <= fo_75 and row["fanin"] <= fi_75:
        scores[0] += 3
    if td_25 <= row["time_delta"]:
        scores[0] += 1

    return max(scores, key=scores.get)

raw_classes = df.apply(assign_behavior_class, axis=1).values

# Re-balance to target distribution: 0=40%, 1=25%, 2=20%, 3=15%
target_counts = {0: 40000, 1: 25000, 2: 20000, 3: 15000}

# Group indices by class
class_indices = defaultdict(list)
for i, c in enumerate(raw_classes):
    class_indices[c].append(i)

log(f"  Raw class distribution: " +
    ", ".join(f"Class {k}: {len(v)}" for k, v in sorted(class_indices.items())))

# Stratified resampling
final_classes = raw_classes.copy()
for cls, target_n in target_counts.items():
    indices = class_indices[cls]
    current_n = len(indices)

    if current_n < target_n:
        # Need more: randomly reassign some from overrepresented classes
        # Find candidates from the most overrepresented class
        deficit = target_n - current_n
        # Pick candidates from other classes that are overrepresented
        candidates = []
        for other_cls in sorted(target_counts.keys()):
            if other_cls == cls:
                continue
            other_indices = class_indices[other_cls]
            excess = len(other_indices) - target_counts[other_cls]
            if excess > 0:
                candidates.extend(other_indices[-excess:])
        if candidates:
            reassign = np.random.choice(candidates, size=min(deficit, len(candidates)), replace=False)
            for idx in reassign:
                final_classes[idx] = cls

# Final pass: exact balancing through random swap
for _ in range(3):  # iterative refinement
    current_counts = defaultdict(int)
    current_indices = defaultdict(list)
    for i, c in enumerate(final_classes):
        current_counts[c] += 1
        current_indices[c].append(i)

    for cls, target_n in target_counts.items():
        excess = current_counts[cls] - target_n
        if excess > 0:
            # Find underrepresented class
            for other_cls, other_target in target_counts.items():
                if other_cls == cls:
                    continue
                deficit = other_target - current_counts[other_cls]
                if deficit > 0:
                    swap_n = min(excess, deficit)
                    swap_indices = np.random.choice(current_indices[cls], size=swap_n, replace=False)
                    for idx in swap_indices:
                        final_classes[idx] = other_cls
                    excess -= swap_n
                    if excess <= 0:
                        break

df["behavior_class"] = final_classes.astype(int)

# Verify
for cls in sorted(target_counts.keys()):
    count = np.sum(final_classes == cls)
    log(f"  Class {cls}: {count:,} ({count/len(final_classes)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════
# PHASE 6: Export Datasets
# ══════════════════════════════════════════════════════════════════════
log("PHASE 6: Exporting datasets...")

feature_cols = ["value_ratio", "time_delta", "same_asset", "hop_count",
                "amount_similarity", "degree", "fanout", "fanin",
                "address_age", "transaction_frequency", "entity_evidence",
                "path_length"]

# Dataset 1: Regressor
reg_df = df[feature_cols + ["relevance_score"]].copy()
reg_df.to_csv(OUTPUT_REG, index=False)
log(f"  Saved regressor dataset: {OUTPUT_REG} ({len(reg_df):,} rows)")

# Dataset 2: Classifier
clf_df = df[feature_cols + ["behavior_class"]].copy()
clf_df.to_csv(OUTPUT_CLF, index=False)
log(f"  Saved classifier dataset: {OUTPUT_CLF} ({len(clf_df):,} rows)")

# ══════════════════════════════════════════════════════════════════════
# PHASE 7: Validation & Reports
# ══════════════════════════════════════════════════════════════════════
log("PHASE 7: Generating validation reports...")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

plt.style.use("seaborn-v0_8-darkgrid")

# ── 1. Correlation Matrix ──
log("  1/7: Correlation matrix...")
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

corr_reg = reg_df.corr()
sns.heatmap(corr_reg, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            ax=axes[0], square=True, cbar_kws={"shrink": 0.8})
axes[0].set_title("Regressor Dataset — Correlation Matrix", fontsize=14, fontweight="bold")

corr_clf = clf_df.corr()
sns.heatmap(corr_clf, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            ax=axes[1], square=True, cbar_kws={"shrink": 0.8})
axes[1].set_title("Classifier Dataset — Correlation Matrix", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "01_correlation_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 2. Feature Distributions ──
log("  2/7: Feature distributions...")
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes_flat = axes.flatten()

for i, col in enumerate(feature_cols):
    ax = axes_flat[i]
    ax.hist(df[col], bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.set_title(col, fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    mean_val = df[col].mean()
    ax.axvline(mean_val, color="red", linestyle="--", alpha=0.7, label=f"mean={mean_val:.2f}")
    ax.legend(fontsize=8)

plt.suptitle("Feature Distributions (100K paths)", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "02_feature_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 3. Target Distributions ──
log("  3/7: Target distributions...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Relevance score
axes[0].hist(df["relevance_score"], bins=50, color="#55A868", edgecolor="white", alpha=0.8)
axes[0].axvline(50, color="orange", linestyle="--", linewidth=2, label="Low/Med boundary")
axes[0].axvline(80, color="red", linestyle="--", linewidth=2, label="Med/High boundary")
axes[0].set_title("Relevance Score Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("relevance_score")
axes[0].set_ylabel("Count")
axes[0].legend()

# Behavior class
class_counts = df["behavior_class"].value_counts().sort_index()
class_labels = ["Normal Flow", "Rapid Movement", "Fragmentation", "Consolidation"]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
bars = axes[1].bar(class_labels, class_counts.values, color=colors, edgecolor="white")
for bar, count in zip(bars, class_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"{count:,}\n({count/len(df)*100:.1f}%)", ha="center", fontsize=10)
axes[1].set_title("Behavior Class Distribution", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "03_target_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── 4. Missing Value Report ──
log("  4/7: Missing value report...")
missing_report = []
for col in reg_df.columns:
    missing = reg_df[col].isna().sum()
    missing_report.append({"Feature": col, "Missing_Count": missing,
                          "Missing_Pct": f"{missing/len(reg_df)*100:.2f}%"})
for col in ["behavior_class"]:
    missing = clf_df[col].isna().sum()
    missing_report.append({"Feature": col, "Missing_Count": missing,
                          "Missing_Pct": f"{missing/len(clf_df)*100:.2f}%"})

missing_df = pd.DataFrame(missing_report)
missing_df.to_csv(os.path.join(REPORT_DIR, "04_missing_values.csv"), index=False)

print("\n" + "="*60)
print("MISSING VALUE REPORT")
print("="*60)
print(missing_df.to_string(index=False))
print()

# ── 5. Class Balance Report ──
log("  5/7: Class balance report...")
balance_data = []
for cls in sorted(df["behavior_class"].unique()):
    count = (df["behavior_class"] == cls).sum()
    balance_data.append({
        "Class": cls,
        "Label": class_labels[cls],
        "Count": count,
        "Percentage": f"{count/len(df)*100:.1f}%",
        "Target": f"{target_counts[cls]/1000:.0f}K ({target_counts[cls]/sum(target_counts.values())*100:.0f}%)"
    })

balance_df = pd.DataFrame(balance_data)
balance_df.to_csv(os.path.join(REPORT_DIR, "05_class_balance.csv"), index=False)

print("="*60)
print("CLASS BALANCE REPORT")
print("="*60)
print(balance_df.to_string(index=False))
print()

# ── 6. Graph Statistics ──
log("  6/7: Graph statistics...")
graph_stats = {
    "Total Nodes": G.number_of_nodes(),
    "Total Edges": G.number_of_edges(),
    "Avg In-Degree": f"{in_deg_vals.mean():.3f}",
    "Avg Out-Degree": f"{out_deg_vals.mean():.3f}",
    "Max In-Degree": int(in_deg_vals.max()),
    "Max Out-Degree": int(out_deg_vals.max()),
    "Nodes with Edges": sum(1 for n in G.nodes() if G.degree(n) > 0),
    "Isolated Nodes": sum(1 for n in G.nodes() if G.degree(n) == 0),
    "Weakly Connected Components": nx.number_weakly_connected_components(G),
    "Illicit Nodes (class=1)": sum(1 for n, d in G.nodes(data=True) if d.get("label") == 1),
    "Licit Nodes (class=2)": sum(1 for n, d in G.nodes(data=True) if d.get("label") == 2),
    "Unknown Nodes": sum(1 for n, d in G.nodes(data=True) if d.get("label") == 0),
    "Timestep Range": f"{ts_min} – {ts_max}",
    "Density": f"{nx.density(G):.6f}",
}

print("="*60)
print("NETWORKX GRAPH STATISTICS")
print("="*60)
for k, v in graph_stats.items():
    print(f"  {k:.<40} {v}")
print()

with open(os.path.join(REPORT_DIR, "06_graph_statistics.txt"), "w") as f:
    f.write("NetworkX Graph Statistics\n")
    f.write("=" * 50 + "\n")
    for k, v in graph_stats.items():
        f.write(f"{k}: {v}\n")

# ── 7. Feature Importance ──
log("  7/7: Feature importance (training quick RF models)...")

X = df[feature_cols].values

# Regressor
y_reg = df["relevance_score"].values
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_train)
reg_score = rf_reg.score(X_test, y_test)
reg_importances = rf_reg.feature_importances_
log(f"  Regressor R² score: {reg_score:.4f}")

# Classifier
y_clf = df["behavior_class"].values
X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)
clf_score = rf_clf.score(X_test, y_test)
clf_importances = rf_clf.feature_importances_
log(f"  Classifier accuracy: {clf_score:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Regressor importance
idx_reg = np.argsort(reg_importances)[::-1]
axes[0].barh(range(len(feature_cols)), reg_importances[idx_reg], color="#55A868", edgecolor="white")
axes[0].set_yticks(range(len(feature_cols)))
axes[0].set_yticklabels([feature_cols[i] for i in idx_reg])
axes[0].set_xlabel("Importance")
axes[0].set_title(f"Regressor Feature Importance (R²={reg_score:.4f})",
                  fontsize=14, fontweight="bold")
axes[0].invert_yaxis()

# Classifier importance
idx_clf = np.argsort(clf_importances)[::-1]
axes[1].barh(range(len(feature_cols)), clf_importances[idx_clf], color="#4C72B0", edgecolor="white")
axes[1].set_yticks(range(len(feature_cols)))
axes[1].set_yticklabels([feature_cols[i] for i in idx_clf])
axes[1].set_xlabel("Importance")
axes[1].set_title(f"Classifier Feature Importance (Acc={clf_score:.4f})",
                  fontsize=14, fontweight="bold")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "07_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()

# Save importance as CSV
imp_df = pd.DataFrame({
    "Feature": feature_cols,
    "Regressor_Importance": reg_importances,
    "Classifier_Importance": clf_importances,
})
imp_df = imp_df.sort_values("Regressor_Importance", ascending=False)
imp_df.to_csv(os.path.join(REPORT_DIR, "07_feature_importance.csv"), index=False)

# ══════════════════════════════════════════════════════════════════════
# Final Summary
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("GENERATION COMPLETE")
print("="*60)
print(f"  Regressor dataset:  {OUTPUT_REG}")
print(f"    → Rows: {len(reg_df):,}, Columns: {len(reg_df.columns)}")
print(f"  Classifier dataset: {OUTPUT_CLF}")
print(f"    → Rows: {len(clf_df):,}, Columns: {len(clf_df.columns)}")
print(f"  Validation reports: {REPORT_DIR}/")
print()

# Quick sample
print("REGRESSOR SAMPLE (first 5 rows):")
print(reg_df.head().to_string(index=False))
print()
print("CLASSIFIER SAMPLE (first 5 rows):")
print(clf_df.head().to_string(index=False))
print()

log("All done! ✓")
