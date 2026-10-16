from __future__ import annotations

from pathlib import Path

import pandas as pd

PREDICTIONS_PATH = Path(
    "data/processed/predictions.parquet"
)

ID_COLUMN = "subscriber_id"
PROBABILITY_COLUMN = "risk_probability"
PREDICTION_COLUMN = "risk_prediction"

EXPECTED_MODEL_NAME = "logistic_regression"
EXPECTED_MODEL_VERSION = "1.0.0"

REQUIRED_COLUMNS = {
    ID_COLUMN,
    PROBABILITY_COLUMN,
    PREDICTION_COLUMN,
    "prediction_timestamp",
    "model_name",
    "model_version",
}


def load_predictions() -> pd.DataFrame:
    """Load the generated prediction dataset."""
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_PATH}"
        )

    predictions = pd.read_parquet(
        PREDICTIONS_PATH
    )

    print(
        f"Loaded predictions: {predictions.shape}"
    )

    return predictions


def validate_columns(
    predictions: pd.DataFrame,
) -> None:
    """Validate the prediction schema."""
    missing_columns = (
        REQUIRED_COLUMNS
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )


def validate_subscribers(
    predictions: pd.DataFrame,
) -> None:
    """Validate subscriber identifiers."""
    if predictions[ID_COLUMN].isna().any():
        raise ValueError(
            "Missing subscriber IDs found."
        )

    if predictions[ID_COLUMN].duplicated().any():
        duplicate_count = int(
            predictions[ID_COLUMN]
            .duplicated()
            .sum()
        )

        raise ValueError(
            "Duplicate subscriber IDs found: "
            f"{duplicate_count}"
        )


def validate_probabilities(
    predictions: pd.DataFrame,
) -> None:
    """Validate risk probabilities."""
    probabilities = predictions[
        PROBABILITY_COLUMN
    ]

    if probabilities.isna().any():
        raise ValueError(
            "Missing risk probabilities found."
        )

    if not pd.api.types.is_numeric_dtype(
        probabilities
    ):
        raise ValueError(
            "Risk probabilities must be numeric."
        )

    invalid_probabilities = (
        ~probabilities.between(0, 1)
    )

    if invalid_probabilities.any():
        count = int(
            invalid_probabilities.sum()
        )

        raise ValueError(
            "Invalid risk probabilities found: "
            f"{count}"
        )


def validate_predictions_values(
    predictions: pd.DataFrame,
) -> None:
    """Validate binary risk predictions."""
    risk_predictions = predictions[
        PREDICTION_COLUMN
    ]

    if risk_predictions.isna().any():
        raise ValueError(
            "Missing risk predictions found."
        )

    invalid_values = ~risk_predictions.isin(
        [0, 1]
    )

    if invalid_values.any():
        values = sorted(
            risk_predictions[
                invalid_values
            ].unique()
        )

        raise ValueError(
            "Invalid risk prediction values found: "
            f"{values}"
        )


def validate_model_metadata(
    predictions: pd.DataFrame,
) -> None:
    """Validate model name and version."""
    model_names = predictions[
        "model_name"
    ].dropna().unique()

    if len(model_names) != 1:
        raise ValueError(
            "Expected exactly one model name, "
            f"found: {model_names.tolist()}"
        )

    if model_names[0] != EXPECTED_MODEL_NAME:
        raise ValueError(
            "Unexpected model name: "
            f"{model_names[0]!r}. "
            f"Expected {EXPECTED_MODEL_NAME!r}."
        )

    model_versions = predictions[
        "model_version"
    ].dropna().unique()

    if len(model_versions) != 1:
        raise ValueError(
            "Expected exactly one model version, "
            f"found: {model_versions.tolist()}"
        )

    if model_versions[0] != EXPECTED_MODEL_VERSION:
        raise ValueError(
            "Unexpected model version: "
            f"{model_versions[0]!r}. "
            f"Expected {EXPECTED_MODEL_VERSION!r}."
        )


def validate_timestamps(
    predictions: pd.DataFrame,
) -> None:
    """Validate prediction timestamps."""
    timestamps = predictions[
        "prediction_timestamp"
    ]

    if timestamps.isna().any():
        raise ValueError(
            "Missing prediction timestamps found."
        )

    parsed_timestamps = pd.to_datetime(
        timestamps,
        errors="coerce",
        utc=True,
    )

    if parsed_timestamps.isna().any():
        invalid_count = int(
            parsed_timestamps.isna().sum()
        )

        raise ValueError(
            "Invalid prediction timestamps found: "
            f"{invalid_count}"
        )


def print_summary(
    predictions: pd.DataFrame,
) -> None:
    """Print prediction validation summary."""
    total_predictions = len(predictions)

    positive_predictions = int(
        predictions[PREDICTION_COLUMN].sum()
    )

    positive_rate = (
        positive_predictions / total_predictions
        if total_predictions
        else 0.0
    )

    average_probability = predictions[
        PROBABILITY_COLUMN
    ].mean()

    print()
    print("Prediction validation passed.")
    print(
        f"Rows: {total_predictions:,}"
    )
    print(
        "Unique subscribers: "
        f"{predictions[ID_COLUMN].nunique():,}"
    )
    print(
        "Predicted positives: "
        f"{positive_predictions:,}"
    )
    print(
        "Predicted positive rate: "
        f"{positive_rate:.4f}"
    )
    print(
        "Average risk probability: "
        f"{average_probability:.4f}"
    )
    print(
        "Model: "
        f"{predictions['model_name'].iloc[0]}"
    )
    print(
        "Model version: "
        f"{predictions['model_version'].iloc[0]}"
    )


def validate_predictions(
    predictions: pd.DataFrame,
) -> None:
    """Run all prediction validation checks."""
    validate_columns(predictions)
    validate_subscribers(predictions)
    validate_probabilities(predictions)
    validate_predictions_values(predictions)
    validate_model_metadata(predictions)
    validate_timestamps(predictions)


def main() -> None:
    """Run prediction dataset validation."""
    print("Loading predictions...")

    predictions = load_predictions()

    validate_predictions(predictions)

    print_summary(predictions)


if __name__ == "__main__":
    main()