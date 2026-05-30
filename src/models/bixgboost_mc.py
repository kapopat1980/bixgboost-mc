"""
BiXGBoost-MC: Full end-to-end pipeline

Implements Algorithm 1 from the paper:
  Phase 1 — BFAL feature augmentation
  Phase 2 — Bi-LSTM temporal encoder training
  Phase 3 — Residual computation
  Phase 4 — XGBoost residual corrector training
  Phase 5 — Final prediction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.features.bfal import BifacialFeatureAugmentationLayer
from src.models.bilstm_encoder import BiLSTMEncoder
from src.models.xgboost_corrector import XGBoostResidualCorrector, build_xrc_features
from src.utils.seed_utils import set_all_seeds


class BiXGBoostMC:
    """
    Full BiXGBoost-MC pipeline.

    Parameters
    ----------
    config : dict
        Hyperparameter config, typically loaded from configs/bixgboost_mc.yaml.
    seed : int
        Random seed for this run.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(self, config: dict, seed: int = 42, device: str = "cpu"):
        self.config = config
        self.seed = seed
        self.device = torch.device(device)
        set_all_seeds(seed)

        bl = config["bilstm"]
        xb = config["xgboost"]
        bf = config["bfal"]

        self.bfal = BifacialFeatureAugmentationLayer(
            tilt_deg=bf["tilt_angle_deg"],
            row_height_m=bf["row_height_m"],
            row_pitch_m=bf["row_pitch_m"],
            wccf_alpha=bf["wccf_alpha"],
            t_ref=bf["t_ref_celsius"],
        )
        self.encoder = BiLSTMEncoder(
            input_dim=13,
            hidden_dims=bl["hidden_dims"],
            forecast_horizon=config["forecasting"]["horizons"][0],
            dropout=bl["dropout"],
        ).to(self.device)

        self.corrector = XGBoostResidualCorrector(
            n_estimators=xb["n_estimators"],
            max_depth=xb["max_depth"],
            learning_rate=xb["learning_rate"],
            subsample=xb["subsample"],
            colsample_bytree=xb["colsample_bytree"],
            reg_lambda=xb["reg_lambda"],
            huber_slope=xb["huber_slope"],
            early_stopping_rounds=xb["early_stopping_rounds"],
        )

        self.scaler_params: dict | None = None   # min-max scaler fit on training set
        self.lookback = bl["lookback_window"]

    # ------------------------------------------------------------------
    # Phase 1 — BFAL
    # ------------------------------------------------------------------
    def augment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.bfal.transform(df)

    # ------------------------------------------------------------------
    # Phase 2 — Bi-LSTM training
    # ------------------------------------------------------------------
    def train_encoder(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> list[float]:
        """Train the Bi-LSTM encoder with Adam + early stopping."""
        cfg = self.config["bilstm"]

        opt = torch.optim.Adam(
            self.encoder.parameters(),
            lr=cfg["learning_rate"],
            betas=tuple(cfg["optimizer"]["betas"]),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            patience=cfg["lr_reduce_patience"],
            factor=cfg["lr_reduce_factor"],
        )
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_v = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=cfg["batch_size"],
            shuffle=True,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        val_losses = []

        for epoch in range(cfg["max_epochs"]):
            self.encoder.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                y_hat, _, _ = self.encoder(xb)
                loss = criterion(y_hat.squeeze(-1), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), cfg["gradient_clip_max_norm"]
                )
                opt.step()

            # Validation
            self.encoder.eval()
            with torch.no_grad():
                y_val_hat, _, _ = self.encoder(X_v)
                val_loss = criterion(y_val_hat.squeeze(-1), y_v).item()
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg["early_stopping_patience"]:
                    break

        return val_losses

    # ------------------------------------------------------------------
    # Phase 3 — Residual computation
    # ------------------------------------------------------------------
    def compute_residuals(
        self, X: np.ndarray, y_true: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (residuals, lstm_predictions, context_vectors)."""
        self.encoder.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_hat, ctx, _ = self.encoder(X_t)
        y_hat_np = y_hat.squeeze(-1).cpu().numpy()
        ctx_np = ctx.cpu().numpy()
        residuals = y_true - y_hat_np
        return residuals, y_hat_np, ctx_np

    # ------------------------------------------------------------------
    # Phase 4 — XRC training
    # ------------------------------------------------------------------
    def train_corrector(
        self,
        Z_train: np.ndarray,
        residuals_train: np.ndarray,
        Z_val: np.ndarray,
        residuals_val: np.ndarray,
    ):
        self.corrector.fit(Z_train, residuals_train, Z_val, residuals_val)

    # ------------------------------------------------------------------
    # Phase 5 — Inference
    # ------------------------------------------------------------------
    def predict(
        self,
        X: np.ndarray,
        bfal_features: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> np.ndarray:
        """
        Full BiXGBoost-MC prediction (Eq. 11).
        """
        _, lstm_pred, ctx = self.compute_residuals(X, np.zeros(len(X)))
        Z = build_xrc_features(bfal_features, ctx, lstm_pred, timestamps)
        return self.corrector.predict(Z, lstm_pred)

    # ------------------------------------------------------------------
    # Convenience: full fit
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        bfal_train: np.ndarray,
        ts_train: pd.DatetimeIndex,
        X_val: np.ndarray,
        y_val: np.ndarray,
        bfal_val: np.ndarray,
        ts_val: pd.DatetimeIndex,
    ) -> "BiXGBoostMC":
        """Run all four training phases in sequence (Algorithm 1)."""
        # Phase 2
        self.train_encoder(X_train, y_train, X_val, y_val)
        # Phase 3
        res_train, lstm_train, ctx_train = self.compute_residuals(X_train, y_train)
        _, lstm_val, ctx_val = self.compute_residuals(X_val, y_val)
        res_val = y_val - lstm_val
        # Phase 4
        Z_train = build_xrc_features(bfal_train, ctx_train, lstm_train, ts_train)
        Z_val = build_xrc_features(bfal_val, ctx_val, lstm_val, ts_val)
        self.train_corrector(Z_train, res_train, Z_val, res_val)
        return self
