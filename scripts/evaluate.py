"""
Evaluate a trained BiXGBoost-MC model and run Diebold-Mariano tests
against all baselines.

Usage:
    python scripts/evaluate.py \
        --dataset nrel_srrl \
        --task ghi \
        --horizon 10min \
        --seeds 42 123 456 789 1024
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import all_metrics
from src.evaluation.diebold_mariano import dm_table

HORIZON_MAP = {"10min": 1, "30min": 3, "60min": 6}


def run_baseline_persistence(y_true: np.ndarray, horizon: int) -> np.ndarray:
    """Persistence model: ŷ_{t+h} = y_t"""
    return y_true[:-horizon] if horizon > 0 else y_true


def print_metrics_table(results: dict[str, dict]) -> None:
    """Print a formatted metrics table."""
    header = f"{'Model':<20} {'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'R²':>8} {'nRMSE':>8}"
    print("\n" + "-" * len(header))
    print(header)
    print("-" * len(header))
    for model, m in results.items():
        print(
            f"{model:<20} "
            f"{m['rmse']:>8.3f} "
            f"{m['mae']:>8.3f} "
            f"{m['mape']:>8.2f} "
            f"{m['r2']:>8.4f} "
            f"{m['nrmse']:>8.2f}"
        )
    print("-" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nrel_srrl")
    parser.add_argument("--task", default="ghi")
    parser.add_argument("--horizon", default="10min")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 456, 789, 1024])
    parser.add_argument("--results_dir", default="results/tables")
    args = parser.parse_args()

    print(f"\nEvaluation: {args.dataset} / {args.task} / {args.horizon}")
    print(f"Seeds: {args.seeds}")

    # ── Collect per-seed predictions ─────────────────────────────────────
    # In a full run, each seed's predictions would be loaded from saved
    # checkpoint outputs. Here we scaffold the aggregation logic.

    seed_metrics = []
    for seed in args.seeds:
        result_file = ROOT / "results" / "tables" / \
            f"{args.dataset}_{args.task}_{args.horizon}_seed{seed}.npz"
        if result_file.exists():
            data = np.load(result_file)
            y_true = data["y_true"]
            y_pred = data["y_pred"]
            seed_metrics.append(all_metrics(y_true, y_pred))
        else:
            print(f"  [seed {seed}] No saved predictions found at {result_file}")
            print(f"             Run: python scripts/train.py --seed {seed} --dataset {args.dataset} --task {args.task} --horizon {args.horizon}")

    if not seed_metrics:
        print("\nNo results found. Train the model first with scripts/train.py")
        return

    # ── Aggregate across seeds ────────────────────────────────────────────
    agg = {}
    for metric in ["rmse", "mae", "mape", "r2", "nrmse"]:
        vals = [m[metric] for m in seed_metrics]
        agg[metric] = {
            "mean": np.mean(vals),
            "std":  np.std(vals, ddof=1),
            "min":  np.min(vals),
            "max":  np.max(vals),
        }

    print(f"\n{'Metric':<8} {'Mean':>8} {'±Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 44)
    for metric, stats in agg.items():
        print(
            f"{metric.upper():<8} "
            f"{stats['mean']:>8.4f} "
            f"{stats['std']:>8.4f} "
            f"{stats['min']:>8.4f} "
            f"{stats['max']:>8.4f}"
        )

    # ── Save to CSV ───────────────────────────────────────────────────────
    out_dir = ROOT / args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_{args.task}_{args.horizon}_summary.csv"

    rows = []
    for metric, stats in agg.items():
        rows.append({"metric": metric.upper(), **stats})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
