import pandas as pd


def audit_features(
    features: pd.DataFrame,
    target_column: str = "risk_label",
) -> pd.DataFrame:
    """Generate feature-level quality and target-separation statistics."""

    if target_column not in features.columns:
        raise ValueError(
            f"Missing target column: {target_column}"
        )

    excluded_columns = {
        "subscriber_id",
        target_column,
    }

    feature_columns = [
        column
        for column in features.columns
        if column not in excluded_columns
    ]

    rows = []

    for column in feature_columns:
        series = features[column]

        row = {
            "feature": column,
            "dtype": str(series.dtype),
            "n_unique": series.nunique(dropna=False),
            "missing_count": series.isna().sum(),
            "missing_rate": series.isna().mean(),
        }

        if pd.api.types.is_numeric_dtype(series):
            row["min"] = series.min()
            row["max"] = series.max()
            row["mean"] = series.mean()
            row["std"] = series.std()

            positive_mean = features.loc[
                features[target_column] == 1,
                column,
            ].mean()

            negative_mean = features.loc[
                features[target_column] == 0,
                column,
            ].mean()

            row["positive_mean"] = positive_mean
            row["negative_mean"] = negative_mean

            if negative_mean != 0:
                row["positive_negative_ratio"] = (
                    positive_mean / negative_mean
                )
            else:
                row["positive_negative_ratio"] = None

        else:
            row["min"] = None
            row["max"] = None
            row["mean"] = None
            row["std"] = None
            row["positive_mean"] = None
            row["negative_mean"] = None
            row["positive_negative_ratio"] = None

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            "feature",
        )
        .reset_index(drop=True)
    )


def find_constant_features(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """Return features with only one unique value."""

    return audit.loc[
        audit["n_unique"] <= 1
    ].copy()


def find_high_missing_features(
    audit: pd.DataFrame,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Return features exceeding the missing-value threshold."""

    return audit.loc[
        audit["missing_rate"] >= threshold
    ].copy()