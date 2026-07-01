# LedgerGuard — Financial Waste & Anomaly Detection Engine

> AI-powered detection of financial waste, fraud, and risky contract language for SMBs and local governments that can't afford enterprise tools.

## What it does

| Module | Detects | Output |
|--------|---------|--------|
| **1 — Transaction Anomalies** | Duplicate payments, price spikes, new/suspicious vendors, off-hours transactions | Risk score 0–100 + plain-English explanation |
| **2 — Contract NLP** | Auto-renewals, uncapped escalation, vague scope, termination fees | Flagged clauses with risk level + description |
| **3 — Spend Forecasting** | Budget overruns 6–12 months ahead | Forecast with confidence intervals + breach alerts |

## Architecture (Module 1)

Three complementary models, ensemble-fused:

```
Isolation Forest (unsupervised)  ──┐
Autoencoder (reconstruction err)  ─┼─► Weighted ensemble → Risk score → Tier (HIGH/MEDIUM/LOW)
LightGBM (supervised, if labels)  ─┘
                                        │
                                   SHAP values → plain-English explanation
```

**Why this ensemble?**
- IF catches global outliers even without labels (procurement data is rarely labelled)
- AE catches contextual anomalies (unusual patterns not global outliers)
- LightGBM dominates when fraud labels exist (creditcard/paysim datasets)
- Alert fatigue addressed via tiering: only HIGH-confidence anomalies alert immediately

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up Kaggle credentials (for dataset download)
# Place kaggle.json in ~/.kaggle/ or set env vars:
#   KAGGLE_USERNAME, KAGGLE_KEY
```

## Quick Start

### Generate sample data + train on it
```bash
# Generate synthetic demo transactions (no Kaggle needed)
python scripts/generate_sample.py

# Train on the demo data
python scripts/train.py --csv sample_data/demo_transactions.csv \
    --save-dir models_saved/demo

# Run inference and see alerts
python scripts/infer.py \
    --csv sample_data/demo_transactions.csv \
    --model-dir models_saved/demo \
    --tier HIGH
```

### Train on creditcard fraud dataset (recommended for best metrics)
```bash
# Download dataset (requires Kaggle credentials)
python scripts/train.py --dataset creditcard \
    --save-dir models_saved/creditcard

# Evaluate and generate EVAL.md
python scripts/evaluate.py \
    --model-dir models_saved/creditcard \
    --dataset creditcard
```

### Launch the demo dashboard
```bash
streamlit run src/ledgerguard/api/app.py
```

### Analyze a contract
```bash
python scripts/analyze_contract.py \
    --file sample_data/sample_contract.txt
```

## File structure

```
ledgerguard/
├── requirements.txt
├── README.md
├── EVAL.md                         # model comparison and metrics
├── data/                           # downloaded datasets (gitignored)
├── models_saved/                   # trained model artifacts
├── sample_data/
│   ├── demo_transactions.csv       # synthetic demo (generated)
│   └── sample_contract.txt         # demo contract with risky clauses
├── src/ledgerguard/
│   ├── data/
│   │   ├── loader.py               # dataset loaders (creditcard, paysim, generic)
│   │   └── preprocessor.py         # feature engineering + vendor dedup
│   ├── models/
│   │   ├── isolation_forest.py     # unsupervised scorer
│   │   ├── autoencoder.py          # reconstruction-error scorer (CUDA-ready)
│   │   ├── lgbm_model.py           # supervised LightGBM scorer
│   │   ├── ensemble.py             # fusion + SHAP + plain-English explanations
│   │   ├── contract_nlp.py         # Module 2: contract risk analysis
│   │   └── forecaster.py           # Module 3: budget forecasting
│   └── api/
│       └── app.py                  # Streamlit demo dashboard
└── scripts/
    ├── train.py                    # end-to-end training script
    ├── infer.py                    # inference on new CSV → JSON alerts
    ├── evaluate.py                 # metrics + plots → EVAL.md
    ├── generate_sample.py          # synthetic demo data generator
    └── analyze_contract.py         # Module 2 CLI
```

## Key design decisions

**Alert fatigue**: only HIGH-risk (score ≥ 70) transactions surface immediately. MEDIUM-risk (40–69) are batched into a daily digest. LOW (<40) are logged only.

**Class imbalance**: LightGBM uses `scale_pos_weight` (neg/pos ratio) + optional SMOTE. IF and AE are inherently imbalance-agnostic.

**Vendor deduplication**: `rapidfuzz` token_sort_ratio at threshold 85 clusters "Bob's Plumbing" and "Bobs Plumbing Svc" before feature engineering, so vendor-level statistics are accurate.

**Explainability**: SHAP TreeExplainer on LightGBM drives the top-feature attribution. Templated English sentences convert SHAP values into readable alerts (e.g., "This $8,500 payment is 167% above this vendor's 18-month average of $3,200").

**Short series forecasting**: Prophet handles seasonality on as few as 12 monthly data points; falls back to ARIMA → Holt-Winters if Prophet unavailable.

## GPU usage

The autoencoder automatically uses CUDA when available (RTX 5060 fully supported). No configuration needed — PyTorch detects the GPU.

## Datasets

| Dataset | Source | Module | Rows | Fraud rate |
|---------|--------|--------|------|------------|
| creditcardfraud | `mlg-ulb/creditcardfraud` | 1 | 284,807 | 0.17% |
| PaySim | `ealaxi/paysim1` | 1 | 6.3M | 0.13% |
| BankSim | `ealaxi/banksim1` | 1 | 594,643 | 1.2% |
| demo_transactions | generated | 1 | ~5,000 | ~3.5% |

See [EVAL.md](EVAL.md) for model performance on each dataset.
