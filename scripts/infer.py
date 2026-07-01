import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgerguard.data.loader import load_generic
from ledgerguard.data.preprocessor import dedupe_vendors, engineer_features, get_feature_matrix
from ledgerguard.models.ensemble import EnsembleScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("infer")


def run_inference(
    csv_path: Path,
    model_dir: Path,
    tier_filter: str = "ALL",
    top_n: int = 0,
) -> list[dict]:
    meta = json.loads((model_dir / "meta.json").read_text())
    input_dim = len(meta["feature_names"])
    scorer = EnsembleScorer.load(model_dir, input_dim)
    scaler = joblib.load(model_dir / "scaler.pkl")

    df = load_generic(csv_path)
    if "vendor" in df.columns:
        df["vendor"], _ = dedupe_vendors(df["vendor"])
    df = engineer_features(df)
    X, _, _ = get_feature_matrix(df, scaler=scaler, fit=False)

    results = scorer.score_batch(X, df)

    if tier_filter != "ALL":
        results = [r for r in results if r.risk_tier == tier_filter]
    results.sort(key=lambda r: r.risk_score, reverse=True)
    if top_n > 0:
        results = results[:top_n]

    output = []
    for r in results:
        row = df.iloc[r.index].to_dict()
        clean_row = {
            k: (None if (isinstance(v, float) and np.isnan(v)) else
                int(v) if isinstance(v, (np.integer,)) else
                float(v) if isinstance(v, (np.floating,)) else v)
            for k, v in row.items()
        }
        entry = asdict(r)
        entry["transaction"] = clean_row
        output.append(entry)

    return output


def main():
    parser = argparse.ArgumentParser(description="LedgerGuard inference")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models_saved/default"))
    parser.add_argument("--out", type=Path, default=Path("results.json"))
    parser.add_argument("--tier", choices=["ALL", "HIGH", "MEDIUM", "LOW"], default="ALL")
    parser.add_argument("--top", type=int, default=0, help="Show top N results")
    args = parser.parse_args()

    results = run_inference(args.csv, args.model_dir, args.tier, args.top)
    args.out.write_text(json.dumps(results, indent=2))

    high = sum(1 for r in results if r["risk_tier"] == "HIGH")
    medium = sum(1 for r in results if r["risk_tier"] == "MEDIUM")
    total_flagged = high + medium
    flagged_amt = sum(
        r["transaction"].get("amount", 0) or 0
        for r in results
        if r["risk_tier"] in ("HIGH", "MEDIUM")
    )

    print(f"\n{'='*55}")
    print(f"  Transactions scored : {len(results):,}")
    print(f"  HIGH-risk alerts    : {high:,}")
    print(f"  MEDIUM-risk (digest): {medium:,}")
    print(f"  Total flagged       : {total_flagged:,}")
    if flagged_amt:
        print(f"  Flagged amount      : ${flagged_amt:,.2f}")
    print(f"{'='*55}")
    print(f"\nResults written: {args.out}")

    if results:
        print(f"\nTop 3 alerts:")
        for r in results[:3]:
            print(f"  [{r['risk_tier']:6s}] score={r['risk_score']:5.1f} - {r['explanation'][:100]}...")


if __name__ == "__main__":
    main()
