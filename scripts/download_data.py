"""
Download all three benchmark datasets from their official public sources.

Usage:
    python scripts/download_data.py --dataset all
    python scripts/download_data.py --dataset nrel_srrl
    python scripts/download_data.py --dataset pvdaq
    python scripts/download_data.py --dataset oedi
"""

import argparse
import os
import sys
from pathlib import Path

RAW_DIR = Path(__file__).parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASET_INFO = {
    "nrel_srrl": {
        "description": "NREL Solar Radiation Research Laboratory BMS (2018–2022)",
        "url": "https://midcdmz.nrel.gov/srrl_bms/",
        "instructions": (
            "1. Go to https://midcdmz.nrel.gov/srrl_bms/\n"
            "2. Select 'Download Data' → choose years 2018–2022\n"
            "3. Select 1-minute resolution, all available fields\n"
            "4. Save CSV files to:  data/raw/nrel_srrl/\n"
            "   Expected files: srrl_bms_YYYY.csv  (one per year)"
        ),
    },
    "pvdaq": {
        "description": "DOE PVDAQ Module Temperature Dataset (2016–2021)",
        "url": "https://pvdaq.nrel.gov/",
        "instructions": (
            "1. Go to https://pvdaq.nrel.gov/\n"
            "2. Select System ID 4 (bifacial PV testbed, Colorado)\n"
            "3. Download 15-minute resolution CSV for 2016–2021\n"
            "4. Save to:  data/raw/pvdaq/\n"
            "   Expected files: pvdaq_system4_YYYY.csv  (one per year)"
        ),
    },
    "oedi": {
        "description": "OEDI Bifacial PV Performance Dataset (2019–2021)",
        "url": "https://oedi-data-lake.s3.amazonaws.com/pvdaq/",
        "instructions": (
            "1. Browse https://data.openei.org/submissions/4568\n"
            "2. Download all CSV files under 'bifacial_field_data/'\n"
            "3. Save to:  data/raw/oedi/\n"
            "   Expected files: bifacial_YYYY_MM.csv"
        ),
    },
}


def print_instructions(name: str) -> None:
    info = DATASET_INFO[name]
    print(f"\n{'='*60}")
    print(f"Dataset: {info['description']}")
    print(f"Source:  {info['url']}")
    print(f"\nManual download steps:")
    print(info["instructions"])
    dest = RAW_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\nDestination folder created: {dest}")
    print("="*60)


def check_existing(name: str) -> bool:
    dest = RAW_DIR / name
    files = list(dest.glob("*.csv")) if dest.exists() else []
    if files:
        print(f"  [{name}] Found {len(files)} existing CSV file(s) in {dest}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Dataset download helper for BiXGBoost-MC"
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "nrel_srrl", "pvdaq", "oedi"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    targets = list(DATASET_INFO.keys()) if args.dataset == "all" else [args.dataset]

    print("\nBiXGBoost-MC — Dataset Download Helper")
    print("All three datasets are freely available from official NREL/DOE sources.")
    print("Registration is NOT required.\n")

    for name in targets:
        if check_existing(name):
            print(f"  [{name}] Skipping — data already present.\n")
        else:
            print_instructions(name)

    print("\nOnce all CSV files are in place, run:")
    print("  python scripts/preprocess.py --dataset all\n")


if __name__ == "__main__":
    main()
