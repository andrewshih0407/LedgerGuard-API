import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from .isolation_forest import IFScorer
from .autoencoder import AEScorer
from .lgbm_model import LGBMScorer

logger = logging.getLogger(__name__)

RISK_HIGH_THRESHOLD = 70
RISK_MEDIUM_THRESHOLD = 40


@dataclass
class ScoredTransaction:
    index: int
    risk_score: float
    risk_tier: str
    if_score: float
    ae_score: float
    lgbm_score: float
    top_features: list[dict]
    explanation: str


class EnsembleScorer:

    def __init__(
        self,
        w_if: float = 0.333,
        w_ae: float = 0.333,
        w_lgbm: float = 0.333,
        high_threshold: float = RISK_HIGH_THRESHOLD,
        medium_threshold: float = RISK_MEDIUM_THRESHOLD,
    ):
        self.w_if = w_if
        self.w_ae = w_ae
        self.w_lgbm = w_lgbm
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

        self.if_scorer: Optional[IFScorer] = None
        self.ae_scorer: Optional[AEScorer] = None
        self.lgbm_scorer: Optional[LGBMScorer] = None
        self.meta_model = None
        self.optimal_threshold: float = 0.5
        self._has_lgbm = False
        self._use_meta = False
        self._lgbm_primary = False
        self._feature_names: list[str] = []
        self._shap_explainer = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
        X_normal: Optional[np.ndarray] = None,
        val_data: Optional[tuple] = None,
        progress: bool = True,
    ) -> "EnsembleScorer":
        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        def log(msg):
            if progress:
                logger.info(msg)

        log("[1/3] Training Isolation Forest …")
        contamination = (
            max(0.001, float(y_train.mean())) if y_train is not None else 0.01
        )
        self.if_scorer = IFScorer(contamination=contamination)
        self.if_scorer.fit(X_train, self._feature_names)

        log("[2/3] Training Autoencoder (normal transactions only) …")
        if X_normal is None:
            X_normal = X_train[y_train == 0] if y_train is not None else X_train
        self.ae_scorer = AEScorer(epochs=30)
        self.ae_scorer.fit(X_normal, self._feature_names)

        if y_train is not None and y_train.sum() >= 10:
            log("[3/3] Training LightGBM (supervised) …")
            self._has_lgbm = True
            self.lgbm_scorer = LGBMScorer(
                use_smote=True,
                n_estimators=600,
                learning_rate=0.03,
                num_leaves=127,
                n_bag=5,
            )
            self.lgbm_scorer.fit(
                X_train, y_train,
                feature_names=self._feature_names,
                eval_set=None,
            )
            try:
                import shap
                self._shap_explainer = shap.TreeExplainer(self.lgbm_scorer._model)
                log("SHAP TreeExplainer ready")
            except Exception as e:
                logger.warning("SHAP setup failed: %s", e)
        else:
            self._has_lgbm = False
            log("[3/3] No labels → skipping LightGBM (unsupervised mode)")

        if self._has_lgbm:
            self._use_meta = False
            self._lgbm_primary = True
            if val_data is not None:
                self._tune_threshold_lgbm(val_data, log)
        elif y_train is not None and y_train.sum() >= 10:
            self._fit_meta(X_train, y_train, val_data, log)
        else:
            self.w_if, self.w_ae, self.w_lgbm = 0.5, 0.5, 0.0
            self._use_meta = False
            self._lgbm_primary = False

        return self

    def _tune_threshold_lgbm(self, val_data, log):
        from sklearn.metrics import precision_recall_curve

        X_val, y_val = val_data
        proba = self.lgbm_scorer.score(X_val) / 100.0
        prec, rec, thresh = precision_recall_curve(y_val, proba)
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = int(np.nanargmax(f1s[:-1])) if len(thresh) else 0
        self.optimal_threshold = float(thresh[best_idx]) if len(thresh) else 0.5
        log(
            f"  tuned LightGBM threshold = {self.optimal_threshold:.4f} "
            f"(val F1={f1s[best_idx]:.4f}, precision={prec[best_idx]:.3f}, recall={rec[best_idx]:.3f})"
        )

    def _base_scores(self, X: np.ndarray) -> np.ndarray:
        cols = [self.if_scorer.score(X), self.ae_scorer.score(X)]
        if self._has_lgbm:
            cols.append(self.lgbm_scorer.score(X))
        return np.column_stack(cols)

    def _fit_meta(self, X_train, y_train, val_data, log):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, precision_recall_curve

        if val_data is not None:
            X_val, y_val = val_data
        else:
            n_val = max(50, int(len(X_train) * 0.2))
            X_val, y_val = X_train[:n_val], y_train[:n_val]

        log("Fitting stacking meta-learner (LogisticRegression) …")
        Z_val = self._base_scores(X_val) / 100.0
        self.meta_model = LogisticRegression(
            class_weight="balanced", max_iter=1000, C=1.0
        )
        self.meta_model.fit(Z_val, y_val)
        self._use_meta = True

        names = ["IsolationForest", "Autoencoder"] + (["LightGBM"] if self._has_lgbm else [])
        coefs = self.meta_model.coef_[0]
        for n, c in zip(names, coefs):
            log(f"  meta weight  {n:16s} = {c:+.3f}")

        probs = self.meta_model.predict_proba(Z_val)[:, 1]
        prec, rec, thresh = precision_recall_curve(y_val, probs)
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = int(np.nanargmax(f1s[:-1])) if len(thresh) else 0
        self.optimal_threshold = float(thresh[best_idx]) if len(thresh) else 0.5
        log(
            f"  tuned threshold = {self.optimal_threshold:.3f} "
            f"(val F1={f1s[best_idx]:.4f})"
        )

    def _combined_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        base = self._base_scores(X)
        if self._lgbm_primary and self._has_lgbm:
            ensemble = base[:, 2]
        elif self._use_meta and self.meta_model is not None:
            proba = self.meta_model.predict_proba(base / 100.0)[:, 1]
            ensemble = proba * 100.0
        else:
            weights = [self.w_if, self.w_ae]
            if self._has_lgbm:
                weights.append(self.w_lgbm)
            weights = np.array(weights) / (np.sum(weights) + 1e-9)
            ensemble = base @ weights
        return ensemble, base

    def _tier(self, score: float) -> str:
        if score >= self.high_threshold:
            return "HIGH"
        if score >= self.medium_threshold:
            return "MEDIUM"
        return "LOW"

    def score_batch(
        self,
        X: np.ndarray,
        df_original: Optional[pd.DataFrame] = None,
    ) -> list[ScoredTransaction]:
        ensemble, base = self._combined_proba(X)
        if_scores = base[:, 0]
        ae_scores = base[:, 1]
        lgbm_scores = base[:, 2] if self._has_lgbm else np.zeros(len(X))

        shap_values = None
        if self._shap_explainer is not None:
            try:
                sv = self._shap_explainer.shap_values(X)
                shap_values = sv[1] if isinstance(sv, list) else sv
            except Exception as e:
                logger.warning("SHAP inference failed: %s", e)

        results = []
        for i in range(len(X)):
            top = self._top_features(X[i], shap_values, i)
            explanation = self._explain(
                ensemble[i], if_scores[i], ae_scores[i], lgbm_scores[i],
                top, df_original.iloc[i] if df_original is not None else None,
            )
            results.append(ScoredTransaction(
                index=i,
                risk_score=round(float(ensemble[i]), 1),
                risk_tier=self._tier(ensemble[i]),
                if_score=round(float(if_scores[i]), 1),
                ae_score=round(float(ae_scores[i]), 1),
                lgbm_score=round(float(lgbm_scores[i]), 1) if self._has_lgbm else -1,
                top_features=top,
                explanation=explanation,
            ))
        return results

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ensemble, _ = self._combined_proba(X)
        return ensemble / 100.0

    def _top_features(self, x_row, shap_values, idx, top_k=3):
        if shap_values is not None:
            sv = shap_values[idx]
            order = np.argsort(np.abs(sv))[::-1][:top_k]
            return [
                {
                    "name": self._feature_names[j],
                    "value": round(float(x_row[j]), 4),
                    "shap": round(float(sv[j]), 4),
                    "direction": "increases_risk" if sv[j] > 0 else "decreases_risk",
                }
                for j in order
            ]
        order = np.argsort(np.abs(x_row))[::-1][:top_k]
        return [
            {
                "name": self._feature_names[j],
                "value": round(float(x_row[j]), 4),
                "shap": None,
                "direction": "high" if x_row[j] > 0 else "low",
            }
            for j in order
        ]

    def _explain(self, score, if_s, ae_s, lgbm_s, top_features, row):
        parts = []
        tier = self._tier(score)
        tier_label = {"HIGH": "high-risk", "MEDIUM": "moderate-risk", "LOW": "low-risk"}[tier]
        parts.append(f"Risk score {score:.0f}/100 ({tier_label}).")

        if row is not None:
            if (
                "amount" in row.index
                and "feat_vendor_mean_amount" in row.index
                and pd.notna(row.get("feat_vendor_mean_amount", np.nan))
            ):
                amt = row["amount"]
                avg = row["feat_vendor_mean_amount"]
                pct = ((amt - avg) / (avg + 1e-6)) * 100
                vendor = row.get("vendor", "this vendor")
                if abs(pct) >= 10:
                    direction = "above" if pct > 0 else "below"
                    parts.append(
                        f"The ${amt:,.2f} payment is {abs(pct):.0f}% {direction} "
                        f"the average of ${avg:,.2f} for {vendor}."
                    )

            if row.get("feat_is_likely_duplicate", 0) == 1:
                parts.append(
                    "This transaction appears to be a potential duplicate — "
                    "the same amount and vendor appear multiple times."
                )
            if row.get("feat_is_new_vendor", 0) == 1:
                vendor = row.get("vendor", "this vendor")
                parts.append(
                    f"{vendor} has fewer than 3 prior transactions on record — "
                    "new or rarely-used vendors carry higher risk."
                )
            if row.get("feat_is_offhours", 0) == 1:
                hour = int(row.get("feat_hour", -1))
                if hour >= 0:
                    parts.append(
                        f"Transaction occurred at {hour:02d}:xx — outside normal business hours."
                    )

        if top_features:
            names = [f["name"].replace("feat_", "").replace("_", " ") for f in top_features[:2]]
            parts.append(f"Top driving factors: {', '.join(names)}.")

        signals = {"Isolation Forest": if_s, "Autoencoder": ae_s}
        if lgbm_s >= 0:
            signals["LightGBM"] = lgbm_s
        agreeing = [k for k, v in signals.items() if v >= self.medium_threshold]
        if len(agreeing) >= 2:
            parts.append(f"{len(agreeing)} of {len(signals)} detection models agree this is suspicious.")

        return " ".join(parts)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self.if_scorer:
            self.if_scorer.save(directory / "if_scorer.pkl")
        if self.ae_scorer:
            self.ae_scorer.save(directory / "ae_scorer.pkl")
        if self.lgbm_scorer:
            self.lgbm_scorer.save(directory / "lgbm_scorer.pkl")
        if self.meta_model is not None:
            joblib.dump(self.meta_model, directory / "meta_model.pkl")
        meta = {
            "w_if": self.w_if, "w_ae": self.w_ae, "w_lgbm": self.w_lgbm,
            "has_lgbm": self._has_lgbm,
            "use_meta": self._use_meta,
            "lgbm_primary": self._lgbm_primary,
            "optimal_threshold": self.optimal_threshold,
            "feature_names": self._feature_names,
            "high_threshold": self.high_threshold,
            "medium_threshold": self.medium_threshold,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("EnsembleScorer saved to %s", directory)

    @classmethod
    def load(cls, directory: Path, input_dim: int) -> "EnsembleScorer":
        meta = json.loads((directory / "meta.json").read_text())
        obj = cls(
            w_if=meta["w_if"], w_ae=meta["w_ae"], w_lgbm=meta["w_lgbm"],
            high_threshold=meta["high_threshold"],
            medium_threshold=meta["medium_threshold"],
        )
        obj._feature_names = meta["feature_names"]
        obj._has_lgbm = meta["has_lgbm"]
        obj._use_meta = meta.get("use_meta", False)
        obj._lgbm_primary = meta.get("lgbm_primary", False)
        obj.optimal_threshold = meta.get("optimal_threshold", 0.5)
        obj.if_scorer = IFScorer.load(directory / "if_scorer.pkl")
        obj.ae_scorer = AEScorer.load(directory / "ae_scorer.pkl", input_dim)
        if obj._has_lgbm:
            obj.lgbm_scorer = LGBMScorer.load(directory / "lgbm_scorer.pkl")
            try:
                import shap
                obj._shap_explainer = shap.TreeExplainer(obj.lgbm_scorer._model)
            except Exception:
                pass
        meta_path = directory / "meta_model.pkl"
        if meta_path.exists():
            obj.meta_model = joblib.load(meta_path)
        logger.info("EnsembleScorer loaded from %s", directory)
        return obj
