"""Build, validate, audit and persist the snapshot feature dataset.

Run with:  python -m scripts.build_features
"""

import json
from pathlib import Path

import pandas as pd

from feature_engineering.builder import build_feature_dataset, feature_manifest
from feature_engineering.validation import validate_feature_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_datasets() -> dict[str, pd.DataFrame]:
    """Load all raw datasets required for feature engineering."""
    return {
        "customers": pd.read_parquet(RAW_DIR / "customers.parquet"),
        "usage": pd.read_parquet(RAW_DIR / "usage_events.parquet"),
        "recharges": pd.read_parquet(RAW_DIR / "recharges.parquet"),
        "network": pd.read_parquet(RAW_DIR / "network_events.parquet"),
        "risk": pd.read_parquet(RAW_DIR / "risk_events.parquet"),
        "snapshot_labels": pd.read_parquet(RAW_DIR / "snapshot_labels.parquet"),
    }


def main() -> None:
    datasets = load_raw_datasets()

    print("Building snapshot feature dataset...")
    features = build_feature_dataset(
        customers=datasets["customers"],
        usage_events=datasets["usage"],
        recharges=datasets["recharges"],
        network_events=datasets["network"],
        risk_events=datasets["risk"],
        snapshot_labels=datasets["snapshot_labels"],
    )

    print(f"Feature dataset shape: {features.shape}")

    print("\nValidating feature dataset...")
    validate_feature_dataset(
        features,
        expected_snapshots=datasets["snapshot_labels"]["snapshot_date"].nunique(),
    )

    manifest = feature_manifest(features)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    (PROCESSED_DIR / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nFeature groups:")
    for group, columns in manifest["groups"].items():
        print(f"  {group:<10} {len(columns):>4} features")

    missing = features.isna().mean()
    print(f"\nColumns with >50% missing: {int((missing > 0.5).sum())}")
    print(f"Rows: {len(features):,}   positive rate: {features['risk_label'].mean():.4f}")
    print(f"Saved to: {PROCESSED_DIR / 'features.parquet'}")


if __name__ == "__main__":
    main()
