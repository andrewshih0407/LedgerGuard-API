"""LedgerGuard Streamlit Demo — screen-record this for the pitch.

Launch:
    streamlit run src/ledgerguard/api/app.py

The app lets you:
  1. Upload a transaction CSV (or use the built-in sample).
  2. Choose which model directory to use.
  3. See a live risk dashboard with plain-English explanations.
  4. Download the flagged transactions as JSON.
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Make src importable
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from ledgerguard.data.loader import load_generic
from ledgerguard.data.preprocessor import dedupe_vendors, engineer_features, get_feature_matrix
from ledgerguard.models.ensemble import EnsembleScorer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LedgerGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_DATA_PATH = ROOT / "sample_data" / "demo_transactions.csv"
MODEL_DIR = ROOT / "models_saved" / "creditcard"


@st.cache_resource(show_spinner="Loading model…")
def load_model(model_dir: str):
    import joblib
    path = Path(model_dir)
    meta = json.loads((path / "meta.json").read_text())
    input_dim = len(meta["feature_names"])
    scorer = EnsembleScorer.load(path, input_dim)
    scaler = joblib.load(path / "scaler.pkl")
    return scorer, scaler


@st.cache_data(show_spinner="Scoring transactions…")
def score_df(df_bytes: bytes, model_dir: str):
    import io, joblib
    path = Path(model_dir)
    scorer, scaler = load_model(model_dir)

    df = pd.read_csv(io.BytesIO(df_bytes))
    df.columns = [c.strip() for c in df.columns]
    if "vendor" in df.columns:
        df["vendor"], _ = dedupe_vendors(df["vendor"])
    df = engineer_features(df)

    X, feat_names, _ = get_feature_matrix(df, scaler=scaler, fit=False)
    results = scorer.score_batch(X, df)
    return df, results


def tier_color(tier: str) -> str:
    return {"HIGH": "#e53935", "MEDIUM": "#fb8c00", "LOW": "#43a047"}.get(tier, "#888")


def tier_emoji(tier: str) -> str:
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(tier, "⚪")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.shields.io/badge/LedgerGuard-v0.1-blue", use_container_width=False)
    st.title("🛡️ LedgerGuard")
    st.caption("Financial Waste & Anomaly Detection")
    st.divider()

    model_dir_input = st.text_input("Model directory", str(MODEL_DIR))

    st.subheader("Upload transactions")
    uploaded = st.file_uploader("CSV file", type=["csv"])

    use_sample = st.checkbox("Use built-in demo data", value=not bool(uploaded))

    st.divider()
    tier_filter = st.multiselect(
        "Show risk tiers",
        ["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM"],
    )
    top_n = st.slider("Max transactions to display", 10, 500, 50)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🛡️ LedgerGuard — Financial Anomaly Detection")
st.caption("AI-powered waste, fraud, and spending-risk detection for SMBs and local governments.")

model_ready = Path(model_dir_input).exists() and (Path(model_dir_input) / "meta.json").exists()

if not model_ready:
    st.warning(
        f"No trained model found at **{model_dir_input}**.\n\n"
        "Run `python scripts/train.py --dataset creditcard --save-dir models_saved/creditcard` first, "
        "then refresh this page."
    )
    st.stop()

# Determine data source
if use_sample or not uploaded:
    if not SAMPLE_DATA_PATH.exists():
        st.error(
            f"Sample data not found at {SAMPLE_DATA_PATH}. "
            "Run `python scripts/generate_sample.py` to create it."
        )
        st.stop()
    df_bytes = SAMPLE_DATA_PATH.read_bytes()
    data_label = "demo_transactions.csv"
else:
    df_bytes = uploaded.read()
    data_label = uploaded.name

with st.spinner("Scoring…"):
    try:
        df, results = score_df(df_bytes, model_dir_input)
    except Exception as e:
        st.error(f"Scoring failed: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Summary KPIs
# ---------------------------------------------------------------------------
high = [r for r in results if r.risk_tier == "HIGH"]
medium = [r for r in results if r.risk_tier == "MEDIUM"]
low = [r for r in results if r.risk_tier == "LOW"]
flagged_amt = sum(
    df["amount"].iloc[r.index] for r in high + medium
    if "amount" in df.columns
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Transactions", f"{len(results):,}")
c2.metric("🔴 HIGH Risk", f"{len(high):,}")
c3.metric("🟡 MEDIUM Risk", f"{len(medium):,}")
c4.metric("🟢 LOW Risk", f"{len(low):,}")
if "amount" in df.columns and flagged_amt > 0:
    c5.metric("💰 Flagged Amount", f"${flagged_amt:,.0f}")

st.divider()

# ---------------------------------------------------------------------------
# Risk score distribution
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Risk Score Distribution")
    scores = [r.risk_score for r in results]
    fig = px.histogram(
        x=scores, nbins=50,
        color_discrete_sequence=["#5c6bc0"],
        labels={"x": "Risk Score (0–100)", "y": "Count"},
    )
    fig.add_vrect(x0=70, x1=100, fillcolor="#e53935", opacity=0.15, line_width=0, annotation_text="HIGH")
    fig.add_vrect(x0=40, x1=70, fillcolor="#fb8c00", opacity=0.12, line_width=0, annotation_text="MEDIUM")
    fig.update_layout(showlegend=False, height=280, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Risk Breakdown")
    pie = go.Figure(go.Pie(
        labels=["HIGH", "MEDIUM", "LOW"],
        values=[len(high), len(medium), len(low)],
        marker_colors=["#e53935", "#fb8c00", "#43a047"],
        hole=0.4,
    ))
    pie.update_layout(height=280, margin=dict(t=20, b=20))
    st.plotly_chart(pie, use_container_width=True)

# ---------------------------------------------------------------------------
# Flagged transactions table
# ---------------------------------------------------------------------------
st.subheader(f"Flagged Transactions — {data_label}")

filtered = [r for r in results if r.risk_tier in tier_filter]
filtered.sort(key=lambda r: r.risk_score, reverse=True)
filtered = filtered[:top_n]

if not filtered:
    st.info("No transactions match the selected risk tiers.")
else:
    rows = []
    for r in filtered:
        row = {
            "Tier": tier_emoji(r.risk_tier) + " " + r.risk_tier,
            "Score": f"{r.risk_score:.0f}",
        }
        if "amount" in df.columns:
            row["Amount"] = f"${df['amount'].iloc[r.index]:,.2f}"
        if "vendor" in df.columns:
            row["Vendor"] = df["vendor"].iloc[r.index]
        if "timestamp" in df.columns:
            row["Date"] = str(df["timestamp"].iloc[r.index])[:10]
        if "category" in df.columns:
            row["Category"] = df["category"].iloc[r.index]
        row["IF"] = f"{r.if_score:.0f}"
        row["AE"] = f"{r.ae_score:.0f}"
        if r.lgbm_score >= 0:
            row["LGBM"] = f"{r.lgbm_score:.0f}"
        rows.append(row)

    table_df = pd.DataFrame(rows)
    st.dataframe(table_df, use_container_width=True, height=min(600, 40 + 35 * len(rows)))

# ---------------------------------------------------------------------------
# Detailed explanation cards
# ---------------------------------------------------------------------------
st.subheader("Alert Details")
n_cards = min(10, len(filtered))
for r in filtered[:n_cards]:
    color = tier_color(r.risk_tier)
    vendor_label = df["vendor"].iloc[r.index] if "vendor" in df.columns else f"Tx #{r.index}"
    amount_label = (
        f"${df['amount'].iloc[r.index]:,.2f}" if "amount" in df.columns else ""
    )
    with st.expander(
        f"{tier_emoji(r.risk_tier)} [{r.risk_tier}] {vendor_label} {amount_label} — Score {r.risk_score:.0f}/100",
        expanded=(r.risk_tier == "HIGH"),
    ):
        st.markdown(f"**Explanation:** {r.explanation}")

        if r.top_features:
            st.markdown("**Top driving features:**")
            feat_df = pd.DataFrame(r.top_features)
            if "shap" in feat_df.columns and feat_df["shap"].notna().any():
                feat_df = feat_df[["name", "value", "shap", "direction"]]
                feat_df.columns = ["Feature", "Value", "SHAP Impact", "Direction"]
            else:
                feat_df = feat_df[["name", "value", "direction"]]
                feat_df.columns = ["Feature", "Value", "Direction"]
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

        cols = st.columns(3)
        cols[0].metric("Isolation Forest", f"{r.if_score:.0f}/100")
        cols[1].metric("Autoencoder", f"{r.ae_score:.0f}/100")
        if r.lgbm_score >= 0:
            cols[2].metric("LightGBM", f"{r.lgbm_score:.0f}/100")

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
st.divider()
export = [asdict(r) for r in filtered]
st.download_button(
    "⬇️ Download flagged transactions (JSON)",
    data=json.dumps(export, indent=2),
    file_name="ledgerguard_alerts.json",
    mime="application/json",
)
