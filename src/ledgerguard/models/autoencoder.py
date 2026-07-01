import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _AE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        dims = [input_dim] + hidden_dims
        encoder_layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            encoder_layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.encoder = nn.Sequential(*encoder_layers)

        rdims = list(reversed(dims))
        decoder_layers: list[nn.Module] = []
        for i in range(len(rdims) - 1):
            act: nn.Module = nn.ReLU() if i < len(rdims) - 2 else nn.Identity()
            decoder_layers += [nn.Linear(rdims[i], rdims[i + 1]), act]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AEScorer:

    def __init__(
        self,
        hidden_dims: Optional[list[int]] = None,
        epochs: int = 30,
        batch_size: int = 2048,
        lr: float = 1e-3,
    ):
        self.hidden_dims = hidden_dims or [64, 32, 16, 32, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._model: Optional[_AE] = None
        self._p95: float = 0.0
        self._p5: float = 0.0
        self.feature_names_: list[str] = []

    def fit(
        self,
        X_train: np.ndarray,
        feature_names: Optional[list[str]] = None,
        val_fraction: float = 0.1,
    ) -> "AEScorer":
        if DEVICE.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("=" * 50)
            logger.info("  AUTOENCODER TRAINING ON GPU: %s", gpu_name)
            logger.info("  %d samples x %d features", *X_train.shape)
            logger.info("=" * 50)
        else:
            logger.info(
                "Training Autoencoder on CPU: %d samples, %d features",
                *X_train.shape,
            )
        self.feature_names_ = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        input_dim = X_train.shape[1]

        self._model = _AE(input_dim, self.hidden_dims).to(DEVICE)
        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        n_val = max(1, int(len(X_train) * val_fraction))
        X_val = X_train[:n_val]
        X_tr = X_train[n_val:]

        tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
        val_tensor = torch.tensor(X_val, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(tr_tensor), batch_size=self.batch_size, shuffle=True
        )

        best_val = float("inf")
        best_state = None
        for epoch in range(self.epochs):
            self._model.train()
            for (batch,) in loader:
                batch = batch.to(DEVICE)
                optimiser.zero_grad()
                loss = criterion(self._model(batch), batch)
                loss.backward()
                optimiser.step()

            self._model.eval()
            with torch.no_grad():
                val_b = val_tensor.to(DEVICE)
                val_loss = criterion(self._model(val_b), val_b).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
            if DEVICE.type == "cuda":
                mem = torch.cuda.memory_allocated(0) / 1024**2
                logger.info(
                    "  epoch %2d/%d  val_mse=%.6f  [GPU mem: %.0f MB]",
                    epoch + 1, self.epochs, val_loss, mem,
                )
            else:
                logger.info("  epoch %2d/%d  val_mse=%.6f", epoch + 1, self.epochs, val_loss)

        if best_state:
            self._model.load_state_dict(best_state)

        errors = self._reconstruction_errors(X_train)
        self._p5 = float(np.percentile(errors, 5))
        self._p95 = float(np.percentile(errors, 99))
        logger.info("AE calibrated: p5=%.4f p99=%.4f", self._p5, self._p95)
        return self

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            recon = self._model(t).cpu().numpy()
        return np.mean((X - recon) ** 2, axis=1)

    def score(self, X: np.ndarray) -> np.ndarray:
        errors = self._reconstruction_errors(X)
        rng = self._p95 - self._p5 + 1e-9
        normalised = (errors - self._p5) / rng * 100
        return np.clip(normalised, 0, 100)

    def save(self, path: Path) -> None:
        import joblib
        joblib.dump(
            {"state_dict": self._model.state_dict(), "meta": self.__dict__ | {"_model": None}},
            path,
        )
        logger.info("AEScorer saved -> %s", path)

    @classmethod
    def load(cls, path: Path, input_dim: int) -> "AEScorer":
        import joblib
        data = joblib.load(path)
        obj = cls()
        meta = data["meta"]
        for k, v in meta.items():
            if k != "_model":
                setattr(obj, k, v)
        obj._model = _AE(input_dim, obj.hidden_dims).to(DEVICE)
        obj._model.load_state_dict(data["state_dict"])
        logger.info("AEScorer loaded <- %s", path)
        return obj
