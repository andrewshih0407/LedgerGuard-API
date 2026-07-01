"""Comprehensive evaluation script — generates EVAL.md metrics.

Usage:
    python scripts/evaluate.py --model-dir models_saved/creditcard --csv data/creditcard/creditcard.csv
    python scripts/evaluate.py --model-dir models_saved/paysim --dataset paysim

Outputs:
  - Console metrics table
  - evaluation_results/ directory with ROC, PR curves, threshold analysis
  - Appends results section to EVAL.md
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgerguard.data.loader import load_creditcard, load_paysim, load_banksim, load_generic
from ledgerguard.data.preprocessor import dedupe_vendors, engineer_features, get_feature_matrix
from ledgerguard.models.ensemble import EnsembleScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")

OUT_DIR = Path("evaluation_results")


def plot_roc(y_true, scores, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    return roc_auc


def plot_pr(y_true, scores, title, out_path):
    prec, rec, thresholds = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    return ap


def threshold_analysis(y_true, scores):
    """Return DataFrame: threshold → precision, recall, F1, FP rate."""
    prec, rec, thresholds = precision_recall_curve(y_true, scores)
    rows = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        preds = (scores >= t).astype(int)
        tn = ((preds == 0) & (y_true == 0)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
            "f1": f1_score(y_true, preds, zero_division=0),
            "fp_rate": fp / (fp + tn + 1e-9),
            "flags": int(preds.sum()),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["creditcard", "paysim", "banksim"])
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--sample", type=int, default=50000)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # Load data
    if args.csv:
        df = load_generic(args.csv)
    elif args.dataset == "creditcard":
        df = load_creditcard()
    elif args.dataset == "paysim":
        df = load_paysim()
    else:
        df = load_banksim()

    if args.sample and len(df) > args.sample:
        # Stratified sample to keep fraud rate
        fraud = df[df["is_fraud"] == 1]
        normal = df[df["is_fraud"] == 0].sample(
            min(args.sample - len(fraud), args.sample), random_state=42
        )
        df = pd.concat([fraud, normal]).sample(frac=1, random_state=42)

    if "vendor" in df.columns:
        df["vendor"], _ = dedupe_vendors(df["vendor"])
    df = engineer_features(df)

    # Load model
    meta = json.loads((args.model_dir / "meta.json").read_text())
    scorer = EnsembleScorer.load(args.model_dir, len(meta["feature_names"]))
    scaler = joblib.load(args.model_dir / "scaler.pkl")
    X, _, _ = get_feature_matrix(df, scaler=scaler, fit=False)

    y = df["is_fraud"].values if "is_fraud" in df.columns else None
    has_labels = y is not None and not np.isnan(y).any()

    # Score
    logger.info("Scoring %d transactions…", len(X))
    results = scorer.score_batch(X, df)
    raw_scores = np.array([r.risk_score / 100 for r in results])

    if not has_labels:
        high = sum(1 for r in results if r.risk_tier == "HIGH")
        medium = sum(1 for r in results if r.risk_tier == "MEDIUM")
        flagged_amt = df["amount"].iloc[
            [i for i, r in enumerate(results) if r.risk_tier in ("HIGH", "MEDIUM")]
        ].sum() if "amount" in df.columns else 0
        print(f"\nUnsupervised mode — no labels available.")
        print(f"  HIGH flags  : {high:,}")
        print(f"  MEDIUM flags: {medium:,}")
        if flagged_amt:
            print(f"  Flagged amt : ${flagged_amt:,.2f}")
        return

    # Supervised metrics
    dataset_name = args.dataset or args.csv.stem if args.csv else "custom"
    roc_auc = plot_roc(y, raw_scores, f"ROC — {dataset_name}", OUT_DIR / "roc.png")
    ap = plot_pr(y, raw_scores, f"Precision-Recall — {dataset_name}", OUT_DIR / "pr.png")
    thresh_df = threshold_analysis(y, raw_scores)

    print(f"\n{'='*60}")
    print(f"EVALUATION — {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"  Samples      : {len(y):,}")
    print(f"  Fraud rate   : {100*y.mean():.4f}%")
    print(f"  AUC-ROC      : {roc_auc:.4f}")
    print(f"  Avg Precision: {ap:.4f}")
    print(f"\nThreshold analysis:")
    print(thresh_df.to_string(index=False, float_format="%.4f"))

    # Best F1 threshold
    best = thresh_df.loc[thresh_df["f1"].idxmax()]
    print(f"\n  Best F1={best['f1']:.4f} at threshold={best['threshold']:.1f}")
    print(f"  FP rate at best F1: {best['fp_rate']:.4f} ({100*best['fp_rate']:.2f}%)")

    high_flags = sum(1 for r in results if r.risk_tier == "HIGH")
    high_true_pos = sum(
        1 for i, r in enumerate(results) if r.risk_tier == "HIGH" and y[i] == 1
    )
    if high_flags:
        print(f"\n  HIGH-risk flags : {high_flags:,}")
        print(f"  True positives  : {high_true_pos:,} ({100*high_true_pos/high_flags:.1f}% precision)")

    # Write metrics to EVAL.md section
    _append_eval_md(dataset_name, roc_auc, ap, best, thresh_df, len(y), y.mean())
    print(f"\nPlots saved → {OUT_DIR}/")


def _append_eval_md(name, roc_auc, ap, best, thresh_df, n, fraud_rate):
    from datetime import date
    section = f"""
## {name.upper()} — {date.today()}

| Metric | Value |
|--------|-------|
| Samples | {n:,} |
| Fraud rate | {100*fraud_rate:.4f}% |
| AUC-ROC | {roc_auc:.4f} |
| Avg Precision (AP) | {ap:.4f} |
| Best F1 | {best['f1']:.4f} (threshold={best['threshold']}) |
| Precision @ best F1 | {best['precision']:.4f} |
| Recall @ best F1 | {best['recall']:.4f} |
| False-Positive Rate @ best F1 | {100*best['fp_rate']:.2f}% |

### Threshold sweep

{thresh_df.to_markdown(index=False, floatfmt=".4f")}

![ROC](evaluation_results/roc.png)
![PR](evaluation_results/pr.png)
"""
    eval_path = Path("EVAL.md")
    existing = eval_path.read_text() if eval_path.exists() else ""
    eval_path.write_text(existing + section)
    logger.info("EVAL.md updated")


if __name__ == "__main__":
    main()
