"""
Train a single BiXGBoost-MC model for a given dataset, task, and horizon.

Usage:
    python scripts/train.py --config configs/bixgboost_mc.yaml \
                            --dataset nrel_srrl \
                            --task ghi \
                            --horizon 10min \
                            --seed 42
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.models.bixgboost_mc import BiXGBoostMC
from src.evaluation.metrics import all_metrics
from src.utils.data_loader import load_dataset, make_sliding_windows
from src.utils.seed_utils import set_all_seeds

HORIZON_MAP = {"10min": 1, "30min": 3, "60min": 6}
TASK_TARGET = {
    "ghi":           "ghi",
    "module_temp":   "t_mod",
    "bifacial_yield":"ghi_poa",
}
BFAL_COLS = ["bgc", "kt", "tri", "hsi", "wccf"]
ALL_FEATURES = [
    "ghi", "dni", "dhi", "t_amb", "rh", "wind_speed", "wind_dir", "p_atm",
    "bgc", "kt", "tri", "hsi", "wccf",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bixgboost_mc.yaml")
    parser.add_argument("--dataset", default="nrel_srrl",
                        choices=["nrel_srrl", "pvdaq", "oedi"])
    parser.add_argument("--task", default="ghi",
                        choices=["ghi", "module_temp", "bifacial_yield"])
    parser.add_argument("--horizon", default="10min",
                        choices=["10min", "30min", "60min"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", default="results/checkpoints")
    args = parser.parse_args()

    with open(ROOT / args.config) as f:
        config = yaml.safe_load(f)

    set_all_seeds(args.seed)
    horizon_steps = HORIZON_MAP[args.horizon]
    config["forecasting"]["horizons"] = [horizon_steps]

    print(f"\nBiXGBoost-MC Training")
    print(f"  Dataset : {args.dataset}")
    print(f"  Task    : {args.task}")
    print(f"  Horizon : {args.horizon} ({horizon_steps} steps)")
    print(f"  Seed    : {args.seed}")
    print(f"  Device  : {args.device}\n")

    # Load data
    train_df = load_dataset(args.dataset, split="train")
    val_df   = load_dataset(args.dataset, split="val")

    target_col = TASK_TARGET[args.task]
    feature_cols = [c for c in ALL_FEATURES if c in train_df.columns]

    lookback = config["bilstm"]["lookback_window"]

    def prepare(df):
        X = df[feature_cols].values
        y = df[target_col].values
        bfal = df[[c for c in BFAL_COLS if c in df.columns]].values
        ts = df.index
        X_seq, y_seq = make_sliding_windows(X, y, lookback, horizon_steps)
        bfal_seq = bfal[lookback + horizon_steps - 1:]
        ts_seq = ts[lookback + horizon_steps - 1:]
        return X_seq, y_seq, bfal_seq, ts_seq

    X_train, y_train, bfal_train, ts_train = prepare(train_df)
    X_val,   y_val,   bfal_val,   ts_val   = prepare(val_df)

    model = BiXGBoostMC(config=config, seed=args.seed, device=args.device)

    t0 = time.time()
    model.fit(X_train, y_train, bfal_train, ts_train,
              X_val,   y_val,   bfal_val,   ts_val)
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min")

    # Validation metrics
    test_df  = load_dataset(args.dataset, split="test")
    X_test, y_test, bfal_test, ts_test = prepare(test_df)
    y_pred = model.predict(X_test, bfal_test, ts_test)
    metrics = all_metrics(y_test, y_pred)
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k.upper():6s}: {v:.4f}")

    # Save checkpoint
    ckpt_dir = ROOT / args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"bilstm_{args.dataset}_{args.task}_{args.horizon}_seed{args.seed}.pt"
    torch.save(model.encoder.state_dict(), ckpt_path)
    print(f"\nEncoder checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
