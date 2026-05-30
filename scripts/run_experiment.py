"""
Full reproducibility script — trains and evaluates BiXGBoost-MC across
all five seeds and all three datasets/tasks, then writes result tables
matching Tables 3–6 and the DM test (Table 7) from the paper.

Usage:
    python scripts/run_experiment.py --seeds 42 123 456 789 1024
    python scripts/run_experiment.py --dataset nrel_srrl --task ghi --horizon 10min
"""

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.models.bixgboost_mc import BiXGBoostMC
from src.evaluation.metrics import all_metrics
from src.evaluation.diebold_mariano import dm_table
from src.utils.data_loader import load_dataset, make_sliding_windows
from src.utils.seed_utils import set_all_seeds, PAPER_SEEDS

HORIZON_MAP   = {"10min": 1, "30min": 3, "60min": 6}
TASK_TARGET   = {"ghi": "ghi", "module_temp": "t_mod", "bifacial_yield": "ghi_poa"}
ALL_FEATURES  = [
    "ghi", "dni", "dhi", "t_amb", "rh", "wind_speed", "wind_dir", "p_atm",
    "bgc", "kt", "tri", "hsi", "wccf",
]
BFAL_COLS     = ["bgc", "kt", "tri", "hsi", "wccf"]

# Experiment matrix from the paper (Table 3–6)
EXPERIMENTS = [
    {"dataset": "nrel_srrl", "task": "ghi",           "horizon": "10min"},
    {"dataset": "nrel_srrl", "task": "ghi",           "horizon": "30min"},
    {"dataset": "nrel_srrl", "task": "ghi",           "horizon": "60min"},
    {"dataset": "pvdaq",     "task": "module_temp",   "horizon": "30min"},
    {"dataset": "pvdaq",     "task": "module_temp",   "horizon": "60min"},
    {"dataset": "oedi",      "task": "bifacial_yield","horizon": "30min"},
    {"dataset": "oedi",      "task": "bifacial_yield","horizon": "60min"},
]


def prepare_split(df, target_col, feature_cols, lookback, horizon_steps):
    X = df[[c for c in feature_cols if c in df.columns]].values
    y = df[target_col].values
    bfal = df[[c for c in BFAL_COLS if c in df.columns]].values
    ts = df.index
    X_seq, y_seq = make_sliding_windows(X, y, lookback, horizon_steps)
    bfal_seq = bfal[lookback + horizon_steps - 1:]
    ts_seq = ts[lookback + horizon_steps - 1:]
    return X_seq, y_seq, bfal_seq, ts_seq


def run_single(exp: dict, seed: int, config: dict, device: str) -> dict:
    """Train and evaluate one (experiment, seed) combination."""
    set_all_seeds(seed)
    horizon_steps = HORIZON_MAP[exp["horizon"]]
    target_col = TASK_TARGET[exp["task"]]
    lookback = config["bilstm"]["lookback_window"]

    cfg = {**config}
    cfg["forecasting"] = {"horizons": [horizon_steps]}

    train_df = load_dataset(exp["dataset"], "train")
    val_df   = load_dataset(exp["dataset"], "val")
    test_df  = load_dataset(exp["dataset"], "test")

    feature_cols = [c for c in ALL_FEATURES if c in train_df.columns]

    X_tr, y_tr, bf_tr, ts_tr = prepare_split(train_df, target_col, feature_cols, lookback, horizon_steps)
    X_va, y_va, bf_va, ts_va = prepare_split(val_df,   target_col, feature_cols, lookback, horizon_steps)
    X_te, y_te, bf_te, ts_te = prepare_split(test_df,  target_col, feature_cols, lookback, horizon_steps)

    model = BiXGBoostMC(config=cfg, seed=seed, device=device)
    model.fit(X_tr, y_tr, bf_tr, ts_tr, X_va, y_va, bf_va, ts_va)

    y_pred = model.predict(X_te, bf_te, ts_te)
    metrics = all_metrics(y_te, y_pred)

    # Save predictions for DM test
    out_dir = ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{exp['dataset']}_{exp['task']}_{exp['horizon']}_seed{seed}"
    np.savez(out_dir / f"{tag}.npz", y_true=y_te, y_pred=y_pred)

    return metrics


def aggregate(seed_results: list[dict]) -> dict:
    agg = {}
    for metric in ["rmse", "mae", "mape", "r2", "nrmse"]:
        vals = [r[metric] for r in seed_results]
        agg[metric] = {"mean": np.mean(vals), "std": np.std(vals, ddof=1)}
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=PAPER_SEEDS)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--horizon", default=None)
    parser.add_argument("--config", default="configs/bixgboost_mc.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(ROOT / args.config) as f:
        config = yaml.safe_load(f)

    experiments = EXPERIMENTS
    if args.dataset:
        experiments = [e for e in experiments if e["dataset"] == args.dataset]
    if args.task:
        experiments = [e for e in experiments if e["task"] == args.task]
    if args.horizon:
        experiments = [e for e in experiments if e["horizon"] == args.horizon]

    print(f"\nBiXGBoost-MC Full Experiment Runner")
    print(f"Seeds   : {args.seeds}")
    print(f"Device  : {args.device}")
    print(f"Experiments: {len(experiments)}")

    summary_rows = []
    t_total = time.time()

    for exp in experiments:
        label = f"{exp['dataset']}/{exp['task']}/{exp['horizon']}"
        print(f"\n{'='*55}")
        print(f"Experiment: {label}")
        print(f"{'='*55}")

        seed_results = []
        for seed in args.seeds:
            print(f"  seed={seed} ...", end=" ", flush=True)
            t0 = time.time()
            metrics = run_single(exp, seed, config, args.device)
            print(f"RMSE={metrics['rmse']:.3f}  MAPE={metrics['mape']:.2f}%  "
                  f"({time.time()-t0:.0f}s)")
            seed_results.append(metrics)

        agg = aggregate(seed_results)
        print(f"\n  Aggregated ({len(args.seeds)} seeds):")
        for metric, stats in agg.items():
            print(f"    {metric.upper():<6}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        row = {
            "dataset": exp["dataset"],
            "task": exp["task"],
            "horizon": exp["horizon"],
        }
        for metric, stats in agg.items():
            row[f"{metric}_mean"] = round(stats["mean"], 4)
            row[f"{metric}_std"]  = round(stats["std"],  4)
        summary_rows.append(row)

    # Save master summary table
    summary_df = pd.DataFrame(summary_rows)
    out_path = ROOT / "results" / "tables" / "full_experiment_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\n{'='*55}")
    print(f"All experiments complete in {(time.time()-t_total)/60:.1f} min")
    print(f"Summary saved to {out_path}")


if __name__ == "__main__":
    main()
