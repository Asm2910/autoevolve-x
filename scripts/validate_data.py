"""Validate the raw v2 datasets.

Run with:  python -m scripts.validate_data
"""

import sys

import pandas as pd

from data_generation.validation import EXPECTED_COLUMNS, validate_raw_datasets
from scripts.generate_data import RAW_DATA_DIR, load_config


def main() -> None:
    config = load_config()
    datasets = {name: pd.read_parquet(RAW_DATA_DIR / f"{name}.parquet") for name in EXPECTED_COLUMNS}

    for name, frame in datasets.items():
        print(f"Loaded {name}: {frame.shape}")

    errors = validate_raw_datasets(datasets, config)
    if errors:
        print("\nValidation FAILED:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    labels = datasets["snapshot_labels"]
    print("\nAll raw dataset checks passed.")
    print(f"Snapshots: {labels['snapshot_date'].nunique()}  rows: {len(labels):,}")
    print(f"Positive rate: {labels['risk_label'].mean():.4f}")


if __name__ == "__main__":
    main()
