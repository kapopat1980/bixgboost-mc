"""
Pre-processing pipeline for all three benchmark datasets.

Steps (Section 3.2 of paper):
  1. Load raw CSVs
  2. Resample to 10-minute resolution
  3. Filter nighttime records (solar elevation < 5°)
  4. Fill short gaps (≤ 30 min) by linear interpolation
  5. Apply BFAL feature augmentation
  6. Chronological 70/10/20 split
  7. Min-max normalisation (fit on training set only)
  8. Save as Parquet to data/processed/

Usage:
    python scripts/preprocess.py --dataset all
    python scripts/preprocess.py --dataset nrel_srrl --verbose
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.features.bfal import BifacialFeatureAugmentationLayer
from src.utils.data_loader import chronological_split, minmax_normalize

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SOLAR_ELEV_THRESHOLD = 5.0    # degrees — nighttime filter
MAX_GAP_STEPS = 3              # max 3 × 10 min = 30 min filled by interpolation

# ── Column name mappings (raw → standardised) ───────────────────────────────

COLUMN_MAP = {
    "nrel_srrl": {
        "Global CMP22 (vent/cor) [W/m^2]":  "ghi",
        "Direct NIP [W/m^2]":               "dni",
        "Diffuse 8-48 (vent/cor) [W/m^2]":  "dhi",
        "Tower Dry Bulb Temp [deg C]":       "t_amb",
        "Tower RH [%]":                      "rh",
        "Avg Wind Speed @ 10m [m/s]":        "wind_speed",
        "Avg Wind Direction @ 10m [deg]":    "wind_dir",
        "Station Pressure [mBar]":           "p_atm",
        "Solar Zenith Angle [degrees]":      "sza",
    },
    "pvdaq": {
        "poa_irradiance":   "ghi_poa",
        "irradiance":       "ghi",
        "module_temp":      "t_mod",
        "wind_speed":       "wind_speed",
        "ambient_temp":     "t_amb",
        "humidity":         "rh",
        "pressure":         "p_atm",
    },
    "oedi": {
        "GHI":              "ghi",
        "DHI":              "dhi",
        "Albedo":           "albedo",
        "T_mod_front":      "t_mod",
        "Wind_speed":       "wind_speed",
        "T_amb":            "t_amb",
        "RH":               "rh",
    },
}

BFAL_CONFIG = {
    "nrel_srrl": dict(tilt_deg=25.0, row_height_m=1.5, row_pitch_m=6.0,
                      wccf_alpha=0.012, t_ref=25.0, ground_albedo=0.20),
    "pvdaq":     dict(tilt_deg=20.0, row_height_m=1.2, row_pitch_m=5.5,
                      wccf_alpha=0.012, t_ref=25.0, ground_albedo=0.25),
    "oedi":      dict(tilt_deg=20.0, row_height_m=1.5, row_pitch_m=5.0,
                      wccf_alpha=0.012, t_ref=25.0, ground_albedo=None),
}

ALL_FEATURES = [
    "ghi", "dni", "dhi", "t_amb", "rh", "wind_speed", "wind_dir", "p_atm",
    "bgc", "kt", "tri", "hsi", "wccf",
]


def load_raw(name: str, verbose: bool = False) -> pd.DataFrame:
    src = RAW_DIR / name
    csvs = sorted(src.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV files in {src}. Run: python scripts/download_data.py --dataset {name}"
        )
    frames = []
    for f in csvs:
        df = pd.read_csv(f, parse_dates=True, index_col=0, low_memory=False)
        frames.append(df)
        if verbose:
            print(f"  Loaded {f.name}: {len(df):,} rows")
    return pd.concat(frames).sort_index()


def resample_10min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to 10-minute means, forward-fill gaps of ≤ MAX_GAP_STEPS."""
    df = df.resample("10T").mean()
    df = df.interpolate(method="linear", limit=MAX_GAP_STEPS)
    return df


def filter_night(df: pd.DataFrame) -> pd.DataFrame:
    """Remove nighttime rows where solar elevation ≤ threshold."""
    if "sza" in df.columns:
        return df[df["sza"] < (90 - SOLAR_ELEV_THRESHOLD)]
    # Fallback: remove rows where GHI ≤ 0
    return df[df["ghi"] > 0]


def run_bfal(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Apply Bifacial Feature Augmentation Layer."""
    cfg = BFAL_CONFIG[dataset]
    bfal = BifacialFeatureAugmentationLayer(**cfg)
    # Ensure required columns exist, filling missing ones with plausible defaults
    if "t_mod" not in df.columns:
        df["t_mod"] = df["t_amb"] + 25.0 * df.get("ghi", 0) / 1000.0  # NOCT approx.
    if "ghi_poa" not in df.columns:
        df["ghi_poa"] = df["ghi"]
    if "dni" not in df.columns:
        df["dni"] = 0.0
    if "dhi" not in df.columns:
        df["dhi"] = 0.0
    if "wind_dir" not in df.columns:
        df["wind_dir"] = 0.0
    if "p_atm" not in df.columns:
        df["p_atm"] = 1013.25
    if "rh" not in df.columns:
        df["rh"] = 50.0
    return bfal.transform(df)


def preprocess_dataset(name: str, verbose: bool = False) -> None:
    print(f"\nProcessing: {name}")

    # 1. Load
    df = load_raw(name, verbose=verbose)

    # 2. Rename columns
    mapping = COLUMN_MAP.get(name, {})
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})

    # 3. Resample
    df = resample_10min(df)

    # 4. Night filter
    n_before = len(df)
    df = filter_night(df)
    if verbose:
        print(f"  Night filter: {n_before:,} → {len(df):,} rows")

    # 5. BFAL
    df = run_bfal(df, name)

    # 6. Drop rows with any NaN in feature columns
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    df = df.dropna(subset=feature_cols)

    # 7. Split
    train, val, test = chronological_split(df, train_frac=0.70, val_frac=0.10)

    # 8. Normalise
    train, val, test, _ = minmax_normalize(
        train.copy(), val.copy(), test.copy(), feature_cols
    )

    # 9. Save
    for split_name, split_df in [("train", train), ("val", val), ("test", test), ("all", df)]:
        out = PROCESSED_DIR / f"{name}_{split_name}.parquet"
        split_df.to_parquet(out)
        if verbose:
            print(f"  Saved {out.name}: {len(split_df):,} rows")

    print(f"  Done. Train={len(train):,}  Val={len(val):,}  Test={len(test):,}")


def main():
    parser = argparse.ArgumentParser(description="BiXGBoost-MC preprocessing pipeline")
    parser.add_argument("--dataset", choices=["all", "nrel_srrl", "pvdaq", "oedi"],
                        default="all")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    targets = ["nrel_srrl", "pvdaq", "oedi"] if args.dataset == "all" else [args.dataset]
    for name in targets:
        preprocess_dataset(name, verbose=args.verbose)

    print("\nPre-processing complete. Run training with:")
    print("  python scripts/train.py --config configs/bixgboost_mc.yaml\n")


if __name__ == "__main__":
    main()
