import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class IFScorer:

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.01,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._raw_min: Optional[float] = None
        self._raw_max: Optional[float] = None
        self.feature_names_: list[str] = []

    def fit(self, X: np.ndarray, feature_names: Optional[list[str]] = None) -> "IFScorer":
        logger.info("Training IsolationForest on %d samples, %d features", *X.shape)
        self.model.fit(X)
        raw = self.model.score_samples(X)
        self._raw_min = float(raw.min())
        self._raw_max = float(raw.max())
        self.feature_names_ = feature_names or [f"f{i}" for i in range(X.shape[1])]
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        raw = self.model.score_samples(X)
        # more negative = more anomalous; invert and scale to [0, 100]
        inverted = -raw
        rng = (-self._raw_min) - (-self._raw_max) + 1e-9
        normalised = (inverted - (-self._raw_max)) / rng * 100
        return np.clip(normalised, 0, 100)

    def predict_flags(self, X: np.ndarray) -> np.ndarray:
        return (self.model.predict(X) == -1).astype(int)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)
        logger.info("IFScorer saved -> %s", path)

    @classmethod
    def load(cls, path: Path) -> "IFScorer":
        obj = joblib.load(path)
        logger.info("IFScorer loaded <- %s", path)
        return obj
