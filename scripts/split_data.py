from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

FEATURES_PATH = (
    PROCESSED_DIR / "features.parquet"
)

TRAIN_PATH = (
    PROCESSED_DIR / "train.parquet"
)

VALIDATION_PATH = (
    PROCESSED_DIR / "validation.parquet"
)

TEST_PATH = (
    PROCESSED_DIR / "test.parquet"
)

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15


def load_features() -> pd.DataFrame:
    """Load the final feature dataset."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURES_PATH}"
        )

    return pd.read_parquet(FEATURES_PATH)


def validate_input(
    features: pd.DataFrame,
) -> None:
    """Validate the input feature dataset."""
    required_columns = {
        "subscriber_id",
        "risk_label",
    }

    missing_columns = (
        required_columns - set(features.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if features["subscriber_id"].duplicated().any():
        raise ValueError(
            "Duplicate subscriber_id values found."
        )

    if features["risk_label"].isna().any():
        raise ValueError(
            "Missing values found in risk_label."
        )

    if features["risk_label"].nunique() != 2:
        raise ValueError(
            "risk_label must contain exactly two classes."
        )


def split_features(
    features: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create stratified train, validation, and test splits."""
    train_validation, test = train_test_split(
        features,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=features["risk_label"],
    )

    validation_fraction = (
        VALIDATION_SIZE
        / (1.0 - TEST_SIZE)
    )

    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction,
        random_state=RANDOM_STATE,
        stratify=train_validation["risk_label"],
    )

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def validate_splits(
    features: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Validate split integrity and prevent subscriber leakage."""
    train_ids = set(train["subscriber_id"])
    validation_ids = set(
        validation["subscriber_id"]
    )
    test_ids = set(test["subscriber_id"])

    if train_ids & validation_ids:
        raise ValueError(
            "Subscriber leakage between train "
            "and validation splits."
        )

    if train_ids & test_ids:
        raise ValueError(
            "Subscriber leakage between train "
            "and test splits."
        )

    if validation_ids & test_ids:
        raise ValueError(
            "Subscriber leakage between validation "
            "and test splits."
        )

    total_split_rows = (
        len(train)
        + len(validation)
        + len(test)
    )

    if total_split_rows != len(features):
        raise ValueError(
            "Split row counts do not match the "
            "original feature dataset."
        )

    for name, dataframe in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        if dataframe["risk_label"].isna().any():
            raise ValueError(
                f"Missing labels found in {name} split."
            )

        if dataframe["subscriber_id"].duplicated().any():
            raise ValueError(
                f"Duplicate subscribers found in "
                f"{name} split."
            )


def save_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Save train, validation, and test datasets."""
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_parquet(
        TRAIN_PATH,
        index=False,
    )

    validation.to_parquet(
        VALIDATION_PATH,
        index=False,
    )

    test.to_parquet(
        TEST_PATH,
        index=False,
    )


def print_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Print dataset split statistics."""
    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    print("\nSplit summary:")

    for name, dataframe in splits.items():
        print(
            f"  {name:<12}"
            f" rows={len(dataframe):>7,}"
            f" columns={len(dataframe.columns):>3}"
            f" positive_rate="
            f"{dataframe['risk_label'].mean():.4f}"
        )

    print("\nPositive class counts:")

    for name, dataframe in splits.items():
        positives = int(
            dataframe["risk_label"].sum()
        )
        print(
            f"  {name:<12}"
            f" positives={positives:>5,}"
        )

    print("\nSubscriber counts:")

    for name, dataframe in splits.items():
        subscribers = dataframe[
            "subscriber_id"
        ].nunique()

        print(
            f"  {name:<12}"
            f" subscribers={subscribers:>7,}"
        )

    print("\nSaved files:")
    print(f"  {TRAIN_PATH}")
    print(f"  {VALIDATION_PATH}")
    print(f"  {TEST_PATH}")


def main() -> None:
    """Create and validate reproducible dataset splits."""
    print("Loading feature dataset...")

    features = load_features()

    print(
        f"Loaded feature dataset: "
        f"{features.shape}"
    )

    validate_input(features)

    print("Input validation passed.")

    train, validation, test = split_features(
        features
    )

    validate_splits(
        features,
        train,
        validation,
        test,
    )

    print("Split validation passed.")

    save_splits(
        train,
        validation,
        test,
    )

    print_summary(
        train,
        validation,
        test,
    )


if __name__ == "__main__":
    main()