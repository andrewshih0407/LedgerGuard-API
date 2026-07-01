import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class LGBMScorer:

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        use_smote: bool = False,
        n_jobs: int = -1,
        n_bag: int = 1,
    ):
        # seed-bagging: average n_bag models to reduce variance
        self.n_bag = n_bag
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            objective="binary",
            metric="auc",
            verbose=-1,
        )
        self.use_smote = use_smote
        self._model = None
        self._models: list = []
        self.feature_names_: list[str] = []
        self._classes_: np.ndarray = np.array([0, 1])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        eval_set: Optional[tuple] = None,
    ) -> "LGBMScorer":
        try:
            import lightgbm as lgb
        except ImportError:
            raise RuntimeError("lightgbm not installed. Run: pip install lightgbm")

        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]
        X_tr, y_tr = X, y

        if self.use_smote and y.mean() < 0.1:
            try:
                from imblearn.over_sampling import SMOTE
                sm = SMOTE(random_state=42)
                X_tr, y_tr = sm.fit_resample(X, y)
                logger.info("SMOTE: %d -> %d samples", len(X), len(X_tr))
            except ImportError:
                logger.warning("imbalanced-learn not installed; skipping SMOTE")

        neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
        spw = neg / max(pos, 1)
        self.params["scale_pos_weight"] = spw
        logger.info(
            "LGBMScorer: n=%d pos=%d (%.3f%%) scale_pos_weight=%.1f",
            len(y_tr), pos, 100 * pos / len(y_tr), spw,
        )

        val = eval_set or (X_tr, y_tr)
        do_early_stop = eval_set is not None
        self._models = []
        for b in range(self.n_bag):
            params = dict(self.params)
            params["random_state"] = self.params["random_state"] + b
            callbacks = [lgb.log_evaluation(0)]
            if do_early_stop:
                callbacks.insert(0, lgb.early_stopping(30, verbose=False))
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[val],
                callbacks=callbacks,
                feature_name=self.feature_names_,
            )
            self._models.append(model)
            if self.n_bag > 1:
                logger.info("  bagged LightGBM %d/%d trained", b + 1, self.n_bag)
        self._model = self._models[0]
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        probas = np.column_stack([m.predict_proba(X)[:, 1] for m in self._models])
        return probas.mean(axis=1) * 100

    def cross_val_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> dict:
        from sklearn.metrics import (
            average_precision_score, f1_score, precision_score,
            recall_score, roc_auc_score,
        )

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        metrics: dict[str, list] = {k: [] for k in ["auc", "ap", "f1", "precision", "recall"]}

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            clone = LGBMScorer(**{k: v for k, v in self.params.items() if k not in ("scale_pos_weight",)})
            clone.fit(X[tr_idx], y[tr_idx], eval_set=(X[val_idx], y[val_idx]))
            proba = clone.score(X[val_idx]) / 100
            pred = (proba >= 0.5).astype(int)
            metrics["auc"].append(roc_auc_score(y[val_idx], proba))
            metrics["ap"].append(average_precision_score(y[val_idx], proba))
            metrics["f1"].append(f1_score(y[val_idx], pred, zero_division=0))
            metrics["precision"].append(precision_score(y[val_idx], pred, zero_division=0))
            metrics["recall"].append(recall_score(y[val_idx], pred, zero_division=0))
            logger.info("  fold %d: AUC=%.4f F1=%.4f", fold + 1, metrics["auc"][-1], metrics["f1"][-1])

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def feature_importance(self) -> dict[str, float]:
        imp = self._model.feature_importances_
        return dict(zip(self.feature_names_, imp.tolist()))

    def save(self, path: Path) -> None:
        joblib.dump(self, path)
        logger.info("LGBMScorer saved -> %s", path)

    @classmethod
    def load(cls, path: Path) -> "LGBMScorer":
        obj = joblib.load(path)
        logger.info("LGBMScorer loaded <- %s", path)
        return obj
