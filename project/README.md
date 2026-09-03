# Blockchain Forensics ML Investigation Platform

A production-ready Machine Learning pipeline for blockchain transaction graph forensic investigation. The platform analyzes transaction paths extracted from a NetworkX transaction graph and provides:

1. **Relevance Score (Regression)** — Continuous score from `0` to `100` ranking investigative leads.
2. **Risk Level (Thresholding)** — `LOW` (< 50), `MEDIUM` (50–79), `HIGH` (≥ 80).
3. **Behavior Classification (Multi-class Classification)** — Categorizes patterns into:
   - `0: Normal Flow`
   - `1: Rapid Movement`
   - `2: Fragmentation`
   - `3: Consolidation`
4. **Ranked Transaction Path & Explainability** — Global feature importance and SHAP analysis.

---

## Project Structure

```text
project/
├── data/
│   ├── forensic_regressor_dataset.csv     # 100,000 path records
│   └── forensic_classifier_dataset.csv    # 100,000 path records
├── models/
│   ├── random_forest_regressor.pkl        # Trained RF Regressor model
│   ├── random_forest_classifier.pkl       # Trained RF Classifier model
│   ├── scaler_regressor.pkl               # StandardScaler for Regressor
│   ├── scaler_classifier.pkl              # StandardScaler for Classifier
│   ├── feature_list.json                  # Input feature order schema
│   ├── model_metadata.json                # Regressor metadata & metrics
│   ├── classifier_metadata.json           # Classifier metadata & metrics
│   └── global_explanation_report.json     # Feature importance rankings
├── notebooks/                             # Exploration & analysis notebooks
├── plots/
│   ├── regressor_feature_importance.png   # RF Regressor feature importance
│   ├── regressor_actual_vs_predicted.png  # Regressor actual vs predicted plot
│   ├── regressor_residual_distribution.png# Regressor residual distribution
│   ├── classifier_feature_importance.png  # RF Classifier feature importance
│   ├── classifier_confusion_matrix.png    # Classifier confusion matrix
│   ├── classifier_class_distribution.png  # Test set class distribution
│   ├── explainability_importance_comparison.png # Regressor vs Classifier importance
│   ├── shap_regressor_summary.png         # SHAP summary plot for Regressor
│   └── shap_classifier_summary.png        # SHAP summary plot for Classifier
├── src/
│   ├── config.py                          # Centralized paths & parameters
│   ├── utils.py                           # Validation, risk logic, plotting helpers
│   ├── train_regressor.py                 # RF Regressor training pipeline
│   ├── train_classifier.py                # RF Classifier training pipeline
│   ├── inference.py                       # Reusable prediction API & CLI demo
│   └── explainability.py                  # Feature importance & SHAP analysis
├── requirements.txt                       # Dependency specification
└── README.md                              # System documentation
```

---

## Input Features Schema (12 Features)

| Feature | Description | Range / Unit |
|---------|-------------|--------------|
| `value_ratio` | Ratio of consecutive transaction amounts along path | `0.0 – 1.0` |
| `time_delta` | Time difference between consecutive hops | Seconds |
| `same_asset` | Whether asset remained unchanged | `0` (No) / `1` (Yes) |
| `hop_count` | Number of hops from target wallet | `1 – 6` |
| `amount_similarity` | Similarity between consecutive transaction amounts | `0.0 – 1.0` |
| `degree` | Average total node degree along path | Integer / Float |
| `fanout` | Average number of outgoing neighbors | Integer / Float |
| `fanin` | Average number of incoming neighbors | Integer / Float |
| `address_age` | Age of address in active period | Days |
| `transaction_frequency` | Transactions per active period | Float |
| `entity_evidence` | Strength of attribution evidence | `0` (Unknown), `1` (Weak), `2` (Medium), `3` (Strong) |
| `path_length` | Total number of nodes in path (`hop_count + 1`) | Integer |

---

## Execution Guide

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Train Regressor Model

```bash
python src/train_regressor.py
```
- Performs 5-Fold GridSearchCV over hyperparameter grid.
- Saves best model to `models/random_forest_regressor.pkl`.
- Evaluates MAE, MSE, RMSE, R² Score, and CV Scores.
- Generates actual vs predicted, residual distribution, and feature importance plots in `plots/`.

### 3. Train Classifier Model

```bash
python src/train_classifier.py
```
- Performs 5-Fold GridSearchCV over hyperparameter grid.
- Saves best model to `models/random_forest_classifier.pkl`.
- Evaluates Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
- Generates confusion matrix, class distribution, and feature importance plots in `plots/`.

### 4. Run Explainability Analysis

```bash
python src/explainability.py
```
- Extracts top feature importance rankings for both models.
- Generates SHAP summary plots and global explanation report JSON in `models/global_explanation_report.json`.

### 5. Run Inference API & Demo

```bash
python src/inference.py
```

Or import in Python code:

```python
from src.inference import ForensicPredictor

predictor = ForensicPredictor()

features = {
    "value_ratio": 0.34,
    "time_delta": 15,
    "same_asset": 1,
    "hop_count": 2,
    "amount_similarity": 0.42,
    "degree": 10,
    "fanout": 6,
    "fanin": 1,
    "address_age": 220,
    "transaction_frequency": 120,
    "entity_evidence": 1,
    "path_length": 3
}

# Combined prediction for a suspect transaction path
result = predictor.predict_transaction_path(features)
print(result)
# Output:
# {
#   "relevance_score": 92.4,
#   "risk_level": "HIGH",
#   "behavior_class": 2,
#   "behavior_label": "Fragmentation"
# }
```

---

## Risk Level Mapping

- **HIGH Risk**: `relevance_score >= 80`
- **MEDIUM Risk**: `50 <= relevance_score < 80`
- **LOW Risk**: `relevance_score < 50`

---

## Behavior Class Mapping

- `0`: **Normal Flow** (Low fanout/fanin, moderate time_delta, stable value_ratio)
- `1`: **Rapid Movement** (Very small time_delta, high transaction frequency)
- `2`: **Fragmentation** (High fanout, splitting funds across wallets)
- `3`: **Consolidation** (High fanin, merging funds into common destination)
