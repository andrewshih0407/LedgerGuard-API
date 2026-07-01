import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgerguard.data.loader import load_creditcard, load_paysim, load_banksim, load_generic
from ledgerguard.data.preprocessor import (
    dedupe_vendors, engineer_features, get_feature_matrix,
)
from ledgerguard.models.ensemble import EnsembleScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def print_gpu_banner():
    try:
        import torch
        print("=" * 60)
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU DETECTED: {name}")
            print(f"  VRAM: {total:.1f} GB | CUDA {torch.version.cuda} | torch {torch.__version__}")
            print("  Autoencoder will train on GPU.")
        else:
            print("  No CUDA GPU detected - training on CPU.")
        print("=" * 60)
    except ImportError:
        print("  PyTorch not installed - autoencoder unavailable.")


def banner(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def evaluate(scorer: EnsembleScorer, X_test: np.ndarray, y_test: np.ndarray, df_test: pd.DataFrame):
    from sklearn.metrics import (
        average_precision_score, classification_report,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    results = scorer.score_batch(X_test, df_test)
    scores = scorer.predict_proba(X_test)

    thr = scorer.optimal_threshold if (scorer._use_meta or scorer._lgbm_primary) else 0.5
    preds = (scores >= thr).astype(int)

    if y_test is not None and y_test.sum() > 0:
        tn = ((preds == 0) & (y_test == 0)).sum()
        fp = ((preds == 1) & (y_test == 0)).sum()
        fp_rate = fp / (fp + tn + 1e-9)

        auc = roc_auc_score(y_test, scores)
        ap = average_precision_score(y_test, scores)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        acc = (preds == y_test).mean()

        banner("FINAL EVALUATION (held-out test set)")
        print(f"  Test samples        : {len(y_test):,}")
        print(f"  Fraud samples       : {int(y_test.sum()):,} ({100*y_test.mean():.3f}%)")
        print(f"  Decision threshold  : {thr:.3f} (tuned on validation)")
        print(f"  -------------------------------------------------------")
        print(f"  AUC-ROC             : {auc:.4f}")
        print(f"  Avg Precision (PR)  : {ap:.4f}")
        print(f"  Accuracy            : {acc:.4f}  ({100*acc:.1f}%)")
        print(f"  Precision           : {prec:.4f}  ({100*prec:.1f}%)")
        print(f"  Recall (fraud caught): {rec:.4f}  ({100*rec:.1f}%)")
        print(f"  F1 Score            : {f1:.4f}  ({100*f1:.1f}%)")
        print(f"  False-Positive Rate : {100*fp_rate:.2f}%")
        high_risk = sum(1 for r in results if r.risk_tier == "HIGH")
        flagged_amt = df_test["amount"].iloc[
            [i for i, r in enumerate(results) if r.risk_tier in ("HIGH", "MEDIUM")]
        ].sum() if "amount" in df_test.columns else 0
        print(f"  HIGH-risk flags     : {high_risk:,}")
        if flagged_amt:
            print(f"  Flagged amount      : ${flagged_amt:,.2f}")
        print("=" * 60)
        print(classification_report(y_test, preds, target_names=["Normal", "Fraud"]))

        goal = "PASS" if f1 >= 0.90 else "below target"
        print(f"  >> F1 target (0.90): {goal}  (achieved {f1:.4f})")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train LedgerGuard anomaly models")
    parser.add_argument("--dataset", choices=["creditcard", "paysim", "banksim"], default="creditcard")
    parser.add_argument("--csv", type=Path, help="Path to a custom CSV file")
    parser.add_argument("--save-dir", type=Path, default=Path("models_saved/default"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--sample", type=int, default=None, help="Subsample N rows for quick tests")
    parser.add_argument("--no-ae", action="store_true", help="Skip autoencoder (faster)")
    args = parser.parse_args()

    print_gpu_banner()

    if args.csv:
        df = load_generic(args.csv)
    elif args.dataset == "creditcard":
        df = load_creditcard()
    elif args.dataset == "paysim":
        df = load_paysim()
    else:
        df = load_banksim()

    if args.sample:
        df = df.sample(args.sample, random_state=42)

    if "vendor" in df.columns:
        df["vendor"], _ = dedupe_vendors(df["vendor"])
    df = engineer_features(df)

    y = df["is_fraud"].values if "is_fraud" in df.columns else None
    has_labels = y is not None and not np.isnan(y).all() and np.nansum(y) >= 10
    if not has_labels:
        y = None
        logger.info("No valid labels found -> unsupervised mode")

    X, feat_names, scaler = get_feature_matrix(df, fit=True)

    banner("DATASET SUMMARY")
    print(f"  Source        : {args.csv or args.dataset}")
    print(f"  Transactions  : {len(df):,}")
    print(f"  Features      : {X.shape[1]}")
    if y is not None:
        print(f"  Fraud labels  : {int(np.nansum(y)):,} ({100*np.nanmean(y):.2f}%)")
    else:
        print("  Labels        : none (unsupervised mode)")

    if y is not None:
        idx = np.arange(len(df))
        X_tr, X_tmp, y_tr, y_tmp, idx_tr, idx_tmp = train_test_split(
            X, y, idx, test_size=0.4, stratify=y, random_state=42,
        )
        X_val, X_te, y_val, y_te, idx_val, idx_te = train_test_split(
            X_tmp, y_tmp, idx_tmp, test_size=0.5, stratify=y_tmp, random_state=42,
        )
        df_te = df.iloc[idx_te].reset_index(drop=True)
        print(f"\n  Split: train={len(X_tr):,}  val={len(X_val):,}  test={len(X_te):,}")
        val_data = (X_val, y_val)
    else:
        X_tr, X_te = X[:int(len(X) * 0.8)], X[int(len(X) * 0.8):]
        y_tr = y_te = None
        df_te = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)
        val_data = None

    banner("TRAINING (live)")
    import time
    t0 = time.time()
    ensemble = EnsembleScorer()
    ensemble.fit(X_tr, y_tr, feature_names=feat_names, val_data=val_data)
    print(f"\n  Total training time: {time.time() - t0:.1f}s")

    evaluate(ensemble, X_te, y_te, df_te)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(args.save_dir)
    import joblib
    joblib.dump(scaler, args.save_dir / "scaler.pkl")
    import json
    (args.save_dir / "feature_names.json").write_text(json.dumps(feat_names))
    logger.info("All artifacts saved to %s", args.save_dir)
    print(f"\nModel saved: {args.save_dir}")


if __name__ == "__main__":
    main()
