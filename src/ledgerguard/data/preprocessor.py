import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


def dedupe_vendors(
    series: pd.Series,
    threshold: int = 85,
    canonical_map: Optional[dict] = None,
) -> tuple[pd.Series, dict]:
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        logger.warning("rapidfuzz not installed; skipping vendor deduplication")
        return series, {}

    if canonical_map is None:
        canonical_map = {}

    unique_names = series.dropna().unique().tolist()
    mapping: dict[str, str] = dict(canonical_map)

    canonicals: list[str] = list(canonical_map.values())
    for name in unique_names:
        if name in mapping:
            continue
        if not canonicals:
            canonicals.append(name)
            mapping[name] = name
            continue
        match, score, _ = process.extractOne(
            name, canonicals, scorer=fuzz.token_sort_ratio
        )
        if score >= threshold:
            mapping[name] = match
        else:
            canonicals.append(name)
            mapping[name] = name

    result = series.map(mapping).fillna(series)
    n_clusters = len(set(mapping.values()))
    logger.info(
        "Vendor dedup: %d raw names -> %d clusters (threshold=%d)",
        len(unique_names),
        n_clusters,
        threshold,
    )
    return result, mapping


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "vendor" in df.columns and "amount" in df.columns:
        grp = df.groupby("vendor")["amount"]
        df["feat_vendor_mean_amount"] = grp.transform("mean")
        df["feat_vendor_std_amount"] = grp.transform("std").fillna(0)
        df["feat_vendor_tx_count"] = grp.transform("count")
        df["feat_amount_vs_vendor_mean"] = (
            (df["amount"] - df["feat_vendor_mean_amount"])
            / (df["feat_vendor_std_amount"] + 1e-6)
        )
        df["feat_amount_pct_above_vendor_mean"] = (
            (df["amount"] - df["feat_vendor_mean_amount"])
            / (df["feat_vendor_mean_amount"] + 1e-6)
        ) * 100

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["feat_hour"] = ts.dt.hour
        df["feat_day_of_week"] = ts.dt.dayofweek
        df["feat_is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        df["feat_month"] = ts.dt.month
        df["feat_is_offhours"] = ((ts.dt.hour < 6) | (ts.dt.hour >= 22)).astype(int)
    elif "step" in df.columns:
        df["feat_hour"] = df["step"] % 24
        df["feat_day_of_week"] = (df["step"] // 24) % 7
        df["feat_is_weekend"] = (df["feat_day_of_week"] >= 5).astype(int)

    if "vendor" in df.columns and "amount" in df.columns:
        dup_key = df[["vendor", "amount"]].astype(str).agg("|".join, axis=1)
        df["feat_duplicate_count"] = dup_key.map(dup_key.value_counts())
        df["feat_is_likely_duplicate"] = (df["feat_duplicate_count"] > 1).astype(int)

    if "amount" in df.columns:
        df["feat_log_amount"] = np.log1p(df["amount"].clip(lower=0))

    if "sender_balance_before" in df.columns and "sender_balance_after" in df.columns:
        df["feat_sender_balance_delta"] = (
            df["sender_balance_before"] - df["sender_balance_after"]
        )
        df["feat_sender_balance_drained"] = (
            (df["sender_balance_after"] == 0) & (df["sender_balance_before"] > 0)
        ).astype(int)

    if "feat_vendor_tx_count" in df.columns:
        df["feat_is_new_vendor"] = (df["feat_vendor_tx_count"] <= 2).astype(int)

    if "category" in df.columns and "amount" in df.columns:
        cgrp = df.groupby("category")["amount"]
        df["feat_cat_mean_amount"] = cgrp.transform("mean")
        df["feat_amount_vs_cat_mean"] = (
            (df["amount"] - df["feat_cat_mean_amount"])
            / (df["feat_cat_mean_amount"] + 1e-6)
        ) * 100

    if "amount" in df.columns:
        amt = df["amount"].fillna(0)
        df["feat_is_round_1k"] = (amt % 1000 == 0).astype(int)
        df["feat_is_round_500"] = (amt % 500 == 0).astype(int)
        mu, sigma = amt.mean(), amt.std() + 1e-6
        df["feat_amount_zscore_global"] = (amt - mu) / sigma
        if "vendor" in df.columns:
            vmax = df.groupby("vendor")["amount"].transform("max")
            df["feat_amount_vs_vendor_max"] = amt / (vmax + 1e-6)

    return df


FEATURE_COLS_CREDITCARD = [
    f"V{i}" for i in range(1, 29)
] + ["Amount", "feat_log_amount"]

FEATURE_COLS_GENERIC = [
    c for c in [
        "feat_log_amount",
        "feat_vendor_mean_amount",
        "feat_vendor_std_amount",
        "feat_vendor_tx_count",
        "feat_amount_vs_vendor_mean",
        "feat_amount_pct_above_vendor_mean",
        "feat_hour",
        "feat_day_of_week",
        "feat_is_weekend",
        "feat_is_offhours",
        "feat_duplicate_count",
        "feat_is_likely_duplicate",
        "feat_sender_balance_delta",
        "feat_sender_balance_drained",
        "feat_is_new_vendor",
        "feat_cat_mean_amount",
        "feat_amount_vs_cat_mean",
        "feat_is_round_1k",
        "feat_is_round_500",
        "feat_amount_zscore_global",
        "feat_amount_vs_vendor_max",
    ]
]


def get_feature_matrix(
    df: pd.DataFrame, scaler: Optional[RobustScaler] = None, fit: bool = False
) -> tuple[np.ndarray, list[str], RobustScaler]:
    if "V1" in df.columns:
        cols = [c for c in FEATURE_COLS_CREDITCARD if c in df.columns]
    else:
        cols = [c for c in FEATURE_COLS_GENERIC if c in df.columns]
        numeric_extras = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in cols and c not in ("is_fraud", "is_flagged_fraud")
            and not c.startswith("feat_")
        ]
        cols = cols + numeric_extras[:20]

    if not cols:
        raise ValueError("No usable feature columns found in DataFrame.")

    X = df[cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    if scaler is None:
        scaler = RobustScaler()
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, cols, scaler
