import io
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from ledgerguard.data.preprocessor import dedupe_vendors, engineer_features, get_feature_matrix
from ledgerguard.models.ensemble import EnsembleScorer

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("ledgerguard.server")

app = FastAPI(title="LedgerGuard API", version="0.1.0")
_EXTRA_ORIGIN = os.getenv("ALLOWED_ORIGIN", "")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:4173",
        "http://127.0.0.1:5173",
        *([_EXTRA_ORIGIN] if _EXTRA_ORIGIN else []),
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

_scorer: Optional[EnsembleScorer] = None
_scaler = None
_model_name: str = ""


def _load_model(model_dir: Path):
    global _scorer, _scaler, _model_name
    meta = json.loads((model_dir / "meta.json").read_text())
    input_dim = len(meta["feature_names"])
    _scorer = EnsembleScorer.load(model_dir, input_dim)
    _scaler = joblib.load(model_dir / "scaler.pkl")
    _model_name = model_dir.name
    logger.info("Model loaded from %s  (input_dim=%d)", model_dir, input_dim)


@app.on_event("startup")
def startup():
    candidates = [
        ROOT / "models_saved" / "demo",
        ROOT / "models_saved" / "creditcard",
    ]
    for p in candidates:
        if p.exists() and (p / "meta.json").exists():
            _load_model(p)
            return
    logger.warning(
        "No trained model found. Run train.py first, then restart the server."
    )


class Transaction(BaseModel):
    vendor: Optional[str] = None
    amount: Optional[float] = None
    category: Optional[str] = None
    timestamp: Optional[str] = None
    account: Optional[str] = None
    description: Optional[str] = None


class AnalyzeRequest(BaseModel):
    transactions: List[Transaction]


class FeatureFactor(BaseModel):
    name: str
    value: str
    direction: str


class AlertResult(BaseModel):
    index: int
    vendor: Optional[str]
    amount: Optional[float]
    category: Optional[str]
    date: Optional[str]
    risk_score: float
    risk_tier: str
    if_score: float
    ae_score: float
    lgbm_score: float
    explanation: str
    factors: List[str]


class AnalyzeResponse(BaseModel):
    model: str
    total: int
    high: int
    medium: int
    low: int
    flagged_amount: float
    alerts: List[AlertResult]


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _scorer is not None,
        "model": _model_name,
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if _scorer is None:
        raise HTTPException(503, "Model not loaded. Run train.py first.")
    if not req.transactions:
        raise HTTPException(400, "No transactions provided.")

    rows = [t.dict() for t in req.transactions]
    df = pd.DataFrame(rows)

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    if "vendor" in df.columns:
        df["vendor"], _ = dedupe_vendors(df["vendor"].fillna("Unknown"))
    df = engineer_features(df)

    try:
        X, _, _ = get_feature_matrix(df, scaler=_scaler, fit=False)
    except Exception as e:
        raise HTTPException(422, f"Feature extraction failed: {e}")

    scored = _scorer.score_batch(X, df)

    alerts = []
    flagged_amt = 0.0
    for r in scored:
        orig = rows[r.index]
        amt = orig.get("amount") or 0.0
        tier = r.risk_tier
        if tier in ("HIGH", "MEDIUM"):
            flagged_amt += amt or 0.0

        factors = [f["name"].replace("feat_", "").replace("_", " ").title()
                   for f in (r.top_features or [])[:4]]

        alerts.append(AlertResult(
            index=r.index,
            vendor=orig.get("vendor"),
            amount=amt if amt else None,
            category=orig.get("category"),
            date=str(orig.get("timestamp") or "")[:10] or None,
            risk_score=round(r.risk_score, 1),
            risk_tier=tier,
            if_score=round(r.if_score, 1),
            ae_score=round(r.ae_score, 1),
            lgbm_score=round(r.lgbm_score, 1),
            explanation=r.explanation,
            factors=factors,
        ))

    alerts.sort(key=lambda a: a.risk_score, reverse=True)

    # if >25% are HIGH on a small dataset, use relative ranking instead
    n = len(alerts)
    raw_high = sum(1 for a in alerts if a.risk_tier == "HIGH")
    if n > 0 and raw_high / n > 0.25:
        scores = [a.risk_score for a in alerts]
        hi_cut = scores[max(0, int(n * 0.10) - 1)]
        med_cut = scores[max(0, int(n * 0.30) - 1)]
        for a in alerts:
            if a.risk_score >= hi_cut:
                a.risk_tier = "HIGH"
            elif a.risk_score >= med_cut:
                a.risk_tier = "MEDIUM"
            else:
                a.risk_tier = "LOW"

    high = sum(1 for a in alerts if a.risk_tier == "HIGH")
    medium = sum(1 for a in alerts if a.risk_tier == "MEDIUM")
    low = sum(1 for a in alerts if a.risk_tier == "LOW")
    flagged_amt = sum(a.amount or 0 for a in alerts if a.risk_tier in ("HIGH", "MEDIUM"))

    return AnalyzeResponse(
        model=_model_name,
        total=len(alerts),
        high=high,
        medium=medium,
        low=low,
        flagged_amount=round(flagged_amt, 2),
        alerts=alerts,
    )
