"""Temporal train / validation / test split by snapshot date.

The earliest snapshots train the model, the next ones tune the threshold, and
the latest ones are used once for the final test - like deploying a model and
scoring the future. Run with:  python -m scripts.split_data
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

TRAIN_SNAPSHOTS = 5
VALIDATION_SNAPSHOTS = 2
TEST_SNAPSHOTS = 2


def split_by_snapshot(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(features["snapshot_date"].unique())
    needed = TRAIN_SNAPSHOTS + VALIDATION_SNAPSHOTS + TEST_SNAPSHOTS
    if len(dates) != needed:
        raise ValueError(f"Expected {needed} snapshots, found {len(dates)}")

    train_dates = dates[:TRAIN_SNAPSHOTS]
    validation_dates = dates[TRAIN_SNAPSHOTS : TRAIN_SNAPSHOTS + VALIDATION_SNAPSHOTS]
    test_dates = dates[TRAIN_SNAPSHOTS + VALIDATION_SNAPSHOTS :]

    def take(selected: list) -> pd.DataFrame:
        return features[features["snapshot_date"].isin(selected)].reset_index(drop=True)

    return take(train_dates), take(validation_dates), take(test_dates)


def validate_splits(features, train, validation, test) -> None:
    if len(train) + len(validation) + len(test) != len(features):
        raise ValueError("Split row counts do not match the feature dataset")

    if not train["snapshot_date"].max() < validation["snapshot_date"].min():
        raise ValueError("Training snapshots must all precede validation snapshots")
    if not validation["snapshot_date"].max() < test["snapshot_date"].min():
        raise ValueError("Validation snapshots must all precede test snapshots")

    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        if frame["risk_label"].sum() < 100:
            raise ValueError(f"{name} split has fewer than 100 positives")


def main() -> None:
    features = pd.read_parquet(FEATURES_PATH)
    print(f"Loaded feature dataset: {features.shape}")

    train, validation, test = split_by_snapshot(features)
    validate_splits(features, train, validation, test)
    print("Split validation passed.")

    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        frame.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
        first = frame["snapshot_date"].min().date()
        last = frame["snapshot_date"].max().date()
        print(
            f"  {name:<11} rows={len(frame):>7,}  positives={int(frame['risk_label'].sum()):>5,}"
            f"  rate={frame['risk_label'].mean():.4f}  snapshots {first} .. {last}"
        )

    overlap = (
        set(train["subscriber_id"]) & set(test["subscriber_id"])
    )
    print(f"\nSubscribers appearing in both train and test: {len(overlap):,} (expected; they are scored at different dates)")


if __name__ == "__main__":
    main()
