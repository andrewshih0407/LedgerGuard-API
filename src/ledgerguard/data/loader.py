import os
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[4] / "data"


def _kaggle_download(dataset: str, dest: Path) -> None:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "kaggle package not installed. Run: pip install kaggle\n"
            "Then set KAGGLE_USERNAME and KAGGLE_KEY env vars or place "
            "~/.kaggle/kaggle.json."
        )
    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", dataset, dest)
    os.system(
        f'kaggle datasets download -d "{dataset}" -p "{dest}" --unzip'
    )


def _openml_download_creditcard(dest: Path) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    logger.info("Downloading credit-card fraud dataset from OpenML (data_id=1597)...")
    bunch = fetch_openml(data_id=1597, as_frame=True, parser="auto")
    df = bunch.frame.copy()
    if "Class" in df.columns:
        df["Class"] = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "creditcard.csv"
    df.to_csv(out, index=False)
    logger.info("Saved local copy -> %s", out)
    return df


def load_creditcard(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = DATA_DIR / "creditcard" / "creditcard.csv"
    if not path.exists():
        try:
            _kaggle_download("mlg-ulb/creditcardfraud", path.parent)
            if not path.exists():
                raise FileNotFoundError
        except Exception:
            logger.info("Kaggle unavailable; using OpenML mirror instead.")
            df = _openml_download_creditcard(path.parent)
            df.columns = [c.strip() for c in df.columns]
            if "Class" in df.columns:
                df = df.rename(columns={"Class": "is_fraud"})
            logger.info("creditcard: %d rows, fraud rate=%.4f", len(df), df["is_fraud"].mean())
            return df
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "Class" in df.columns:
        df = df.rename(columns={"Class": "is_fraud"})
    logger.info(
        "creditcard: %d rows, fraud rate=%.4f",
        len(df),
        df["is_fraud"].mean(),
    )
    return df


_PAYSIM_COLS = {
    "step": "step",
    "type": "tx_type",
    "amount": "amount",
    "nameOrig": "sender",
    "oldbalanceOrg": "sender_balance_before",
    "newbalanceOrig": "sender_balance_after",
    "nameDest": "receiver",
    "oldbalanceDest": "receiver_balance_before",
    "newbalanceDest": "receiver_balance_after",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
}


def load_paysim(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or DATA_DIR / "paysim" / "PS_20174392719_1491204439457_log.csv"
    if not path.exists():
        _kaggle_download("ealaxi/paysim1", path.parent)
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in _PAYSIM_COLS.items() if k in df.columns})
    logger.info(
        "paysim: %d rows, fraud rate=%.4f", len(df), df["is_fraud"].mean()
    )
    return df


def load_banksim(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or DATA_DIR / "banksim" / "bs140513_032310.csv"
    if not path.exists():
        _kaggle_download("ealaxi/banksim1", path.parent)
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "fraud" in df.columns:
        df = df.rename(columns={"fraud": "is_fraud"})
    logger.info(
        "banksim: %d rows, fraud rate=%.4f", len(df), df["is_fraud"].mean()
    )
    return df


_FRAUD_ALIASES = ["is_fraud", "fraud", "isFraud", "label", "Class", "class", "target"]
_AMOUNT_ALIASES = ["amount", "Amount", "amt", "transaction_amount", "value"]
_VENDOR_ALIASES = [
    "vendor", "merchant", "Merchant", "merchant_name", "nameDest",
    "receiver", "payee", "counterparty",
]
_TIME_ALIASES = [
    "timestamp", "date", "datetime", "Time", "time", "step",
    "transaction_date", "tx_date",
]
_CATEGORY_ALIASES = [
    "category", "Category", "type", "tx_type", "merchant_category",
    "mcc", "sector",
]


def _first_match(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    for a in aliases:
        if a in df.columns:
            return a
    return None


def load_generic(path: Path, label_col: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    lcol = label_col or _first_match(df, _FRAUD_ALIASES)
    if lcol and lcol != "is_fraud":
        df = df.rename(columns={lcol: "is_fraud"})
    elif not lcol:
        df["is_fraud"] = np.nan

    acol = _first_match(df, _AMOUNT_ALIASES)
    if acol and acol != "amount":
        df = df.rename(columns={acol: "amount"})

    vcol = _first_match(df, _VENDOR_ALIASES)
    if vcol and vcol != "vendor":
        df = df.rename(columns={vcol: "vendor"})

    tcol = _first_match(df, _TIME_ALIASES)
    if tcol and tcol != "timestamp":
        df = df.rename(columns={tcol: "timestamp"})

    ccol = _first_match(df, _CATEGORY_ALIASES)
    if ccol and ccol != "category":
        df = df.rename(columns={ccol: "category"})

    logger.info("generic CSV: %d rows from %s", len(df), path)
    return df
