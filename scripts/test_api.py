from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

DATA_PATH = Path(
    "data/processed/test.parquet"
)

API_URL = (
    "http://127.0.0.1:8000/predict"
)

ID_COLUMN = "subscriber_id"
TARGET_COLUMN = "risk_label"

EXCLUDED_COLUMNS = {
    ID_COLUMN,
    TARGET_COLUMN,
}


def load_test_row() -> tuple[str, dict]:
    """Load one real subscriber and its model features."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {DATA_PATH}"
        )

    data = pd.read_parquet(DATA_PATH)

    if data.empty:
        raise ValueError(
            "Test dataset is empty."
        )

    row = data.iloc[0]

    subscriber_id = str(
        row[ID_COLUMN]
    )

    features = {
        column: row[column]
        for column in data.columns
        if column not in EXCLUDED_COLUMNS
    }

    return subscriber_id, features


def make_json_safe(
    features: dict,
) -> dict:
    """Convert pandas and NumPy values into JSON-safe values."""
    safe_features = {}

    for column, value in features.items():
        if pd.isna(value):
            safe_features[column] = None
        elif hasattr(value, "item"):
            safe_features[column] = value.item()
        else:
            safe_features[column] = value

    return safe_features


def call_prediction_api(
    subscriber_id: str,
    features: dict,
) -> dict:
    """Call the prediction endpoint."""
    payload = {
        "subscriber_id": subscriber_id,
        "features": make_json_safe(features),
    }

    response = requests.post(
        API_URL,
        json=payload,
        timeout=30,
    )

    if not response.ok:
        raise RuntimeError(
            "Prediction API request failed.\n"
            f"Status code: {response.status_code}\n"
            f"Response: {response.text}"
        )

    return response.json()


def main() -> None:
    """Run an end-to-end API prediction test."""
    print("Loading test subscriber...")

    subscriber_id, features = load_test_row()

    print(
        f"Subscriber: {subscriber_id}"
    )

    print(
        f"Feature count: {len(features)}"
    )

    print()
    print("Calling prediction API...")

    result = call_prediction_api(
        subscriber_id,
        features,
    )

    print()
    print("Prediction response:")

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    required_fields = {
        "subscriber_id",
        "risk_probability",
        "risk_prediction",
        "model_name",
        "model_version",
        "threshold",
    }

    missing_fields = (
        required_fields
        - set(result)
    )

    if missing_fields:
        raise ValueError(
            "Missing response fields: "
            f"{sorted(missing_fields)}"
        )

    probability = float(
        result["risk_probability"]
    )

    prediction = int(
        result["risk_prediction"]
    )

    if not 0 <= probability <= 1:
        raise ValueError(
            "Risk probability must be between 0 and 1."
        )

    if prediction not in {0, 1}:
        raise ValueError(
            "Risk prediction must be 0 or 1."
        )

    print()
    print("API validation passed.")


if __name__ == "__main__":
    main()