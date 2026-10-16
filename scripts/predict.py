from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

DATA_PATH = Path("data/processed/test.parquet")
MODEL_PATH = Path(
    "models/artifacts/baseline_logistic_regression.joblib"
)
OUTPUT_PATH = Path("data/processed/predictions.parquet")

MODEL_NAME = "logistic_regression"
MODEL_VERSION = "1.0.0"
THRESHOLD = 0.90

TARGET_COLUMN = "risk_label"
ID_COLUMN = "subscriber_id"


def load_data() -> pd.DataFrame:
    """Load the feature dataset used for inference."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {DATA_PATH}"
        )

    data = pd.read_parquet(DATA_PATH)

    if ID_COLUMN not in data.columns:
        raise ValueError(
            f"Missing required column: {ID_COLUMN}"
        )

    return data


def load_model():
    """Load the trained model artifact."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Remove identifiers and target from inference features."""
    excluded_columns = [
        ID_COLUMN,
        TARGET_COLUMN,
    ]

    return data.drop(
        columns=[
            column
            for column in excluded_columns
            if column in data.columns
        ]
    )


def generate_predictions(
    model,
    features: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Generate probabilities and binary predictions."""
    probabilities = model.predict_proba(features)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return (
        pd.Series(probabilities, index=features.index),
        pd.Series(predictions, index=features.index),
    )


def build_prediction_output(
    data: pd.DataFrame,
    probabilities: pd.Series,
    predictions: pd.Series,
) -> pd.DataFrame:
    """Build the production prediction dataset."""
    timestamp = datetime.now(
        UTC
    ).isoformat()

    return pd.DataFrame(
        {
            "subscriber_id": data[ID_COLUMN].values,
            "risk_probability": probabilities.values,
            "risk_prediction": predictions.values,
            "prediction_timestamp": timestamp,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
        }
    )


def validate_predictions(
    predictions: pd.DataFrame,
) -> None:
    """Validate the generated prediction dataset."""
    required_columns = {
        "subscriber_id",
        "risk_probability",
        "risk_prediction",
        "prediction_timestamp",
        "model_name",
        "model_version",
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing prediction columns: "
            f"{sorted(missing_columns)}"
        )

    if predictions["subscriber_id"].duplicated().any():
        raise ValueError(
            "Duplicate subscriber IDs found."
        )

    if predictions["risk_probability"].isna().any():
        raise ValueError(
            "Missing risk probabilities found."
        )

    if not predictions["risk_probability"].between(
        0,
        1,
    ).all():
        raise ValueError(
            "Risk probabilities must be between 0 and 1."
        )

    if not predictions["risk_prediction"].isin(
        [0, 1]
    ).all():
        raise ValueError(
            "Risk predictions must contain only 0 or 1."
        )


def main() -> None:
    """Run production-style batch inference."""
    print("Loading inference dataset...")

    data = load_data()

    print(
        f"Loaded inference dataset: {data.shape}"
    )

    print("Loading model...")

    model = load_model()

    print("Model loaded.")

    features = prepare_features(data)

    print(
        f"Inference features: {features.shape}"
    )

    print(
        f"Using frozen threshold: {THRESHOLD:.2f}"
    )

    probabilities, predictions = (
        generate_predictions(
            model,
            features,
        )
    )

    output = build_prediction_output(
        data,
        probabilities,
        predictions,
    )

    validate_predictions(output)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("Prediction summary:")
    print(
        f"  Subscribers: {len(output):,}"
    )
    print(
        "  Predicted positives: "
        f"{output['risk_prediction'].sum():,}"
    )
    print(
        "  Predicted positive rate: "
        f"{output['risk_prediction'].mean():.4f}"
    )
    print(
        "  Average risk probability: "
        f"{output['risk_probability'].mean():.4f}"
    )

    print()
    print(
        f"Saved predictions to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()