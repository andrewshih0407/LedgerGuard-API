"""Module 2 — Contract risk analysis CLI.

Usage:
    python scripts/analyze_contract.py --file sample_data/sample_contract.txt
    python scripts/analyze_contract.py --file contract.pdf --out risks.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledgerguard.models.contract_nlp import ContractAnalyzer
from dataclasses import asdict


def main():
    parser = argparse.ArgumentParser(description="Analyze contract for risky clauses")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--transformer", action="store_true", help="Use zero-shot NLI model")
    args = parser.parse_args()

    analyzer = ContractAnalyzer(use_transformer=args.transformer)
    findings = analyzer.analyze_file(args.file)

    print(analyzer.report(findings))

    if args.out:
        data = [asdict(f) for f in findings]
        args.out.write_text(json.dumps(data, indent=2))
        print(f"\nJSON saved → {args.out}")

    # Summary
    high = sum(1 for f in findings if f.risk_level == "HIGH")
    medium = sum(1 for f in findings if f.risk_level == "MEDIUM")
    print(f"\n{'='*40}")
    print(f"  HIGH-risk clauses  : {high}")
    print(f"  MEDIUM-risk clauses: {medium}")
    print(f"  Total flagged      : {len(findings)}")


if __name__ == "__main__":
    main()
