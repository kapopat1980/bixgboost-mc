"""
Data loading utilities for the three benchmark datasets.

Downloads from official public sources (see paper Section 3.1):
  - NREL SRRL BMS:     https://midcdmz.nrel.gov/srrl_bms/
  - DOE PVDAQ:         https://pvdaq.nrel.gov/
  - OEDI Bifacial PV:  https://oedi-data-lake.s3.amazonaws.com/pvdaq/
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parents[3] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[3] / "data" / "processed"

DATASET_URLS = {
    "nrel_srrl": "https://midcdmz.nrel.gov/srrl_bms/",
    "pvdaq":     "https://pvdaq.nrel.gov/",
    "oedi":      "https://oedi-data-lake.s3.amazonaws.com/pvdaq/",
}

REQUIRED_COLUMNS = {
    "nrel_srrl": ["ghi", "dni", "dhi", "t_amb", "rh", "wind_speed", "wind_dir", "p_atm"],
    "pvdaq":     ["ghi", "ghi_poa_front", "ghi_poa_rear", "t_mod", "wind_speed", "t_amb"],
    "oedi":      ["ghi", "dhi", "albedo", "t_mod_front", "t_mod_rear", "wind_speed", "t_amb"],
}


def load_dataset(name: str, split: str = "all") -> pd.DataFrame:
    """
    Load a pre-processed dataset from disk.

    Parameters
    ----------
    name : {'nrel_srrl', 'pvdaq', 'oedi'}
    split : {'train', 'val', 'test', 'all'}

    Returns
    -------
    pd.DataFrame with DatetimeIndex at 10-minute resolution.
    """
    path = PROCESSED_DIR / f"{name}_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Pre-processed file not found: {path}\n"
            f"Run:  python scripts/preprocess.py --dataset {name}"
        )
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological 70/10/20 split with no shuffling (Section 3.2).
    """
    N = len(df)
    n_train = int(N * train_frac)
    n_val = int(N * (train_frac + val_frac))
    return df.iloc[:n_train], df.iloc[n_train:n_val], df.iloc[n_val:]


def minmax_normalize(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Min-max normalisation fitted on the training set only (Section 3.2).
    Returns normalised splits and scaler params for inverse transform.
    """
    scaler = {}
    for col in columns:
        col_min = train[col].min()
        col_max = train[col].max()
        denom = col_max - col_min if col_max > col_min else 1.0
        scaler[col] = {"min": col_min, "max": col_max, "denom": denom}
        for df in [train, val, test]:
            df[col] = (df[col] - col_min) / denom
    return train, val, test, scaler


def inverse_transform(
    values: np.ndarray, col: str, scaler: dict
) -> np.ndarray:
    """Reverse min-max normalisation for a single column."""
    s = scaler[col]
    return values * s["denom"] + s["min"]


def make_sliding_windows(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create (input sequence, target) pairs with given lookback window.

    Parameters
    ----------
    X : ndarray (N, features)
    y : ndarray (N,)
    lookback : int   L in paper (default 24 steps = 4 hours)
    horizon : int    K in paper (1, 3, or 6 steps)

    Returns
    -------
    X_seq : ndarray (N - lookback - horizon + 1, lookback, features)
    y_seq : ndarray (N - lookback - horizon + 1,)
    """
    n = len(X) - lookback - horizon + 1
    X_seq = np.stack([X[i:i + lookback] for i in range(n)])
    y_seq = np.array([y[i + lookback + horizon - 1] for i in range(n)])
    return X_seq, y_seq
