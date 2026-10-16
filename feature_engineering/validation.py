import pandas as pd


def validate_feature_dataset(
    features: pd.DataFrame,
) -> None:
    """Validate the final subscriber-level feature dataset."""

    errors = []

    if "subscriber_id" not in features.columns:
        errors.append("Missing subscriber_id column")

    if "risk_label" not in features.columns:
        errors.append("Missing risk_label column")

    if features["subscriber_id"].duplicated().any():
        errors.append("Duplicate subscriber_id values found")

    if features["subscriber_id"].isna().any():
        errors.append("Missing subscriber_id values found")

    if features["risk_label"].isna().any():
        errors.append("Missing risk_label values found")

    if not features["risk_label"].isin([0, 1]).all():
        errors.append("risk_label must contain only 0 and 1")

    numeric_columns = features.select_dtypes(
        include=["number"]
    ).columns

    infinite_counts = (
        features[numeric_columns]
        .replace([float("inf"), float("-inf")], pd.NA)
        .isna()
        .sum()
    )

    infinite_columns = infinite_counts[
        infinite_counts > 0
    ].index.tolist()

    if infinite_columns:
        errors.append(
            "Infinite or invalid numeric values found in: "
            f"{infinite_columns}"
        )

    if features.isna().any().any():
        missing_columns = (
            features.isna()
            .sum()
            .loc[lambda x: x > 0]
            .index.tolist()
        )

        errors.append(
            "Missing values found in: "
            f"{missing_columns}"
        )

    if errors:
        raise ValueError(
            "Feature validation failed:\n- "
            + "\n- ".join(errors)
        )

    print("Feature validation passed.")
    print(f"Rows: {len(features):,}")
    print(f"Columns: {features.shape[1]:,}")
    print(
        "Unique subscribers: "
        f"{features['subscriber_id'].nunique():,}"
    )
    print(
        "Positive rate: "
        f"{features['risk_label'].mean():.4f}"
    )