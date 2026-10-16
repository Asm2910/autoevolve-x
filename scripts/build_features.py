from pathlib import Path

import pandas as pd

from feature_engineering.audit import (
    audit_features,
    find_constant_features,
    find_high_missing_features,
)
from feature_engineering.builder import (
    build_feature_dataset,
    fill_missing_features,
)
from feature_engineering.validation import validate_feature_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_datasets() -> dict[str, pd.DataFrame]:
    """Load all raw datasets required for feature engineering."""
    return {
        "customers": pd.read_parquet(
            RAW_DIR / "customers.parquet"
        ),
        "usage": pd.read_parquet(
            RAW_DIR / "usage_events.parquet"
        ),
        "recharges": pd.read_parquet(
            RAW_DIR / "recharges.parquet"
        ),
        "network": pd.read_parquet(
            RAW_DIR / "network_events.parquet"
        ),
        "risk": pd.read_parquet(
            RAW_DIR / "risk_events.parquet"
        ),
        "labels": pd.read_parquet(
            RAW_DIR / "labels.parquet"
        ),
    }


def main() -> None:
    """Build, validate, audit, and persist the feature dataset."""
    datasets = load_raw_datasets()

    print("Building feature dataset...")

    features = build_feature_dataset(
    customers=datasets["customers"],
    usage_events=datasets["usage"],
    recharges=datasets["recharges"],
    network_events=datasets["network"],
    risk_events=datasets["risk"],
    labels=datasets["labels"],
)

    features = fill_missing_features(features)

    print(f"Feature dataset shape: {features.shape}")

    print("\nValidating feature dataset...")
    validate_feature_dataset(features)

    print("\nRunning feature audit...")
    audit = audit_features(features)

    constant_features = find_constant_features(audit)
    high_missing_features = find_high_missing_features(audit)

    print(f"Total features: {len(audit)}")
    print(
        f"Constant features: "
        f"{len(constant_features)}"
    )
    print(
        f"High-missing features: "
        f"{len(high_missing_features)}"
    )

    if not constant_features.empty:
        print("\nConstant features:")
        print(
            constant_features[
                ["feature", "n_unique"]
            ].to_string(index=False)
        )

    if not high_missing_features.empty:
        print("\nHigh-missing features:")
        print(
            high_missing_features[
                ["feature", "missing_rate"]
            ].to_string(index=False)
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PROCESSED_DIR / "features.parquet"
    )

    features.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"\nSaved feature dataset to: {output_path}"
    )

    print("\nFeature dataset summary:")
    print(f"  Rows: {len(features):,}")
    print(
        f"  Columns: {len(features.columns):,}"
    )
    print(
        "  Unique subscribers: "
        f"{features['subscriber_id'].nunique():,}"
    )
    print(
        "  Missing values: "
        f"{features.isna().sum().sum():,}"
    )
    print(
        "  Positive rate: "
        f"{features['risk_label'].mean():.4f}"
    )


if __name__ == "__main__":
    main()