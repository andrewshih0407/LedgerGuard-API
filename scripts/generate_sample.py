"""Generate realistic synthetic transaction data for the demo.

Produces sample_data/demo_transactions.csv with ~5,000 transactions
including injected anomalies:
  - Duplicate payments (same vendor + amount)
  - Off-hours transactions
  - Price spikes (>150% of vendor average)
  - New/one-off vendors
  - Round-number suspicious amounts

Run: python scripts/generate_sample.py
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "sample_data"
OUT_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)

VENDORS = [
    "Acme Office Supplies", "City Water Authority", "Metro Electric Co.",
    "Premier Janitorial Services", "Tech Solutions LLC", "Riverside Printing",
    "Lakefront Catering", "Allied Security Systems", "Greenfield Landscaping",
    "Central IT Services", "Budget Staffing Inc.", "Regional Gas & Fuel",
    "Horizon Consulting Group", "Valley Plumbing", "Summit Law Offices",
    "Blue Sky Marketing", "Pacific Payroll Partners", "Delta Insurance",
    "Urban Transit Authority", "Pine Ridge Construction",
]

CATEGORIES = [
    "Utilities", "Office Supplies", "IT Services", "Facilities",
    "Professional Services", "Marketing", "Payroll", "Insurance",
    "Maintenance", "Consulting",
]

CATEGORY_BUDGETS = {
    "Utilities": (800, 3000),
    "Office Supplies": (200, 1500),
    "IT Services": (1000, 8000),
    "Facilities": (500, 4000),
    "Professional Services": (2000, 15000),
    "Marketing": (500, 5000),
    "Payroll": (5000, 50000),
    "Insurance": (1000, 8000),
    "Maintenance": (300, 2500),
    "Consulting": (1500, 12000),
}

VENDOR_CATEGORY = {v: random.choice(CATEGORIES) for v in VENDORS}

def make_normal_tx(n: int) -> pd.DataFrame:
    vendors = RNG.choice(VENDORS, n)
    categories = [VENDOR_CATEGORY[v] for v in vendors]
    amounts = []
    for cat in categories:
        lo, hi = CATEGORY_BUDGETS[cat]
        amounts.append(round(float(RNG.uniform(lo, hi)), 2))

    dates = pd.date_range("2023-01-01", periods=n, freq="2h")
    hours = RNG.integers(8, 18, n)
    timestamps = [
        pd.Timestamp(d.date()) + pd.Timedelta(hours=int(h), minutes=int(RNG.integers(0, 60)))
        for d, h in zip(dates, hours)
    ]
    return pd.DataFrame({
        "timestamp": timestamps,
        "vendor": vendors,
        "category": categories,
        "amount": amounts,
        "account": RNG.choice(["General Fund", "Capital Projects", "Operations", "IT Budget"], n),
        "is_fraud": 0,
    })


def _row_to_dict(row: pd.Series) -> dict:
    return row.to_dict()


def inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    anomalies = []

    # 1. Duplicate payments — same vendor + amount, different dates
    for _ in range(40):
        orig = df.sample(1).iloc[0]
        dup = _row_to_dict(orig)
        dup["timestamp"] = orig["timestamp"] + pd.Timedelta(days=int(RNG.integers(1, 7)))
        dup["is_fraud"] = 1
        anomalies.append(dup)

    # 2. Price spikes (>150% of typical)
    for _ in range(60):
        orig = _row_to_dict(df.sample(1).iloc[0])
        orig["amount"] = round(orig["amount"] * float(RNG.uniform(2.5, 5.0)), 2)
        orig["is_fraud"] = 1
        anomalies.append(orig)

    # 3. Off-hours transactions (midnight–5am)
    for _ in range(50):
        orig = _row_to_dict(df.sample(1).iloc[0])
        ts = pd.Timestamp("2023-06-15") + pd.Timedelta(
            hours=int(RNG.integers(0, 4)), minutes=int(RNG.integers(0, 60))
        )
        orig["timestamp"] = ts
        orig["is_fraud"] = 1
        anomalies.append(orig)

    # 4. New/suspicious vendors
    suspicious_vendors = [
        "FastCash Services", "QuickPay LLC", "Anonymous Contractor",
        "Global Ventures Intl", "Premier Ghost Inc.",
    ]
    for v in suspicious_vendors:
        for _ in range(RNG.integers(1, 4)):
            row = {
                "timestamp": pd.Timestamp("2023-07-01") + pd.Timedelta(days=int(RNG.integers(0, 90))),
                "vendor": v,
                "category": "Professional Services",
                "amount": round(float(RNG.uniform(3000, 25000)), 2),
                "account": "General Fund",
                "is_fraud": 1,
            }
            anomalies.append(row)

    # 5. Round-number suspicious payments
    for _ in range(30):
        round_amt = int(RNG.choice([5000, 10000, 25000, 50000, 100000]))
        row = {
            "timestamp": pd.Timestamp("2023-09-01") + pd.Timedelta(days=int(RNG.integers(0, 60))),
            "vendor": RNG.choice(VENDORS),
            "category": "Consulting",
            "amount": float(round_amt),
            "account": "Capital Projects",
            "is_fraud": 1,
        }
        anomalies.append(row)

    anom_df = pd.DataFrame(anomalies)
    combined = pd.concat([df, anom_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined


def main():
    print("Generating synthetic transactions…")
    df = make_normal_tx(4800)
    df = inject_anomalies(df)
    out = OUT_DIR / "demo_transactions.csv"
    df.to_csv(out, index=False)

    fraud_rate = df["is_fraud"].mean()
    total_fraud_amt = df[df["is_fraud"] == 1]["amount"].sum()
    print(f"  Total rows   : {len(df):,}")
    print(f"  Fraud rows   : {df['is_fraud'].sum():,} ({100*fraud_rate:.1f}%)")
    print(f"  Fraud amount : ${total_fraud_amt:,.2f}")
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
