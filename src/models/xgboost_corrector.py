"""
XGBoost Residual Corrector (XRC)

Trained on Bi-LSTM in-sample residuals to learn structured non-linear
correction patterns, as described in Section 4.3 (Equations 9–11).

Input feature vector z_t (148-dim):
    - 13 BFAL features
    - 128 Bi-LSTM context vector (c_t from attended Bi-LSTM)
    - 1 primary Bi-LSTM prediction (ŷ_LSTM_t)
    - 6 temporal calendar features (sin/cos of hour-of-day, day-of-year,
      plus raw hour and doy for tree splits)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold


def build_xrc_features(
    bfal_features: np.ndarray,
    context_vectors: np.ndarray,
    lstm_predictions: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Construct the 148-dimensional XRC input feature vector z_t.

    Parameters
    ----------
    bfal_features : ndarray (N, 13)
    context_vectors : ndarray (N, 128)   attended Bi-LSTM context vectors
    lstm_predictions : ndarray (N,)      primary Bi-LSTM predictions
    timestamps : DatetimeIndex (N,)

    Returns
    -------
    Z : ndarray (N, 148)
    """
    N = len(lstm_predictions)

    # Temporal calendar features (6 cols)
    hour = timestamps.hour.values + timestamps.minute.values / 60.0
    doy = timestamps.dayofyear.values
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    doy_sin = np.sin(2 * np.pi * doy / 365.0)
    doy_cos = np.cos(2 * np.pi * doy / 365.0)
    calendar = np.stack([hour_sin, hour_cos, doy_sin, doy_cos, hour, doy], axis=1)

    Z = np.concatenate(
        [
            bfal_features,               # 13
            context_vectors,             # 128
            lstm_predictions[:, None],   # 1
            calendar,                    # 6
        ],
        axis=1,
    )
    assert Z.shape[1] == 148, f"Expected 148 features, got {Z.shape[1]}"
    return Z


class XGBoostResidualCorrector:
    """
    XGBoost model trained to predict Bi-LSTM residuals (Eq. 9–10).

    Final prediction (Eq. 11): ŷ = ŷ_LSTM + XRC(z_t)

    Parameters
    ----------
    n_estimators : int
    max_depth : int
    learning_rate : float
    subsample : float
    colsample_bytree : float
    reg_lambda : float
    huber_slope : float    delta parameter of pseudo-Huber loss
    early_stopping_rounds : int
    cv_folds : int
        Number of folds for cross-validation on validation set.
    """

    def __init__(
        self,
        n_estimators: int = 800,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.80,
        colsample_bytree: float = 0.80,
        reg_lambda: float = 1.0,
        huber_slope: float = 1.35,
        early_stopping_rounds: int = 50,
        cv_folds: int = 5,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            objective="reg:pseudohubererror",
            huber_slope=huber_slope,
            tree_method="hist",
            verbosity=0,
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.cv_folds = cv_folds
        self.model: xgb.XGBRegressor | None = None

    def fit(
        self,
        Z_train: np.ndarray,
        residuals_train: np.ndarray,
        Z_val: np.ndarray,
        residuals_val: np.ndarray,
    ) -> "XGBoostResidualCorrector":
        """
        Train XRC on (training-set) Bi-LSTM residuals.

        Parameters
        ----------
        Z_train : ndarray (N_train, 148)
        residuals_train : ndarray (N_train,)   ε = y_true − ŷ_LSTM  (Eq. 9)
        Z_val : ndarray (N_val, 148)
        residuals_val : ndarray (N_val,)
        """
        self.model = xgb.XGBRegressor(
            **self.params,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self.model.fit(
            Z_train, residuals_train,
            eval_set=[(Z_val, residuals_val)],
            verbose=False,
        )
        return self

    def predict_correction(self, Z: np.ndarray) -> np.ndarray:
        """Return residual correction Δ̂ for feature matrix Z."""
        if self.model is None:
            raise RuntimeError("XRC has not been trained. Call .fit() first.")
        return self.model.predict(Z)

    def predict(
        self, Z: np.ndarray, lstm_predictions: np.ndarray
    ) -> np.ndarray:
        """
        Final BiXGBoost-MC prediction (Eq. 11).
        ŷ = ŷ_LSTM + XRC(z_t)
        """
        correction = self.predict_correction(Z)
        return lstm_predictions + correction

    def get_feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        """Return XGBoost feature importance scores."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return self.model.get_booster().get_score(importance_type=importance_type)
