"""Send one real test row to a running API and check the response.

Start the API first:  python -m uvicorn api.app:app --port 8000
Then run:             python -m scripts.test_api
"""

import json
import sys

import pandas as pd
import requests

API_URL = "http://localhost:8000"
METADATA_PATH = "models/artifacts/model_metadata.json"
TEST_PATH = "data/processed/test.parquet"

EXPECTED_FIELDS = {
    "subscriber_id",
    "risk_probability",
    "risk_prediction",
    "threshold",
    "model_name",
    "model_version",
    "request_id",
}


def main() -> None:
    with open(METADATA_PATH, encoding="utf-8") as file:
        feature_names = json.load(file)["features"]

    row = pd.read_parquet(TEST_PATH).iloc[[0]]
    payload = {
        "subscriber_id": str(row["subscriber_id"].iloc[0]),
        "features": json.loads(row[feature_names].to_json(orient="records"))[0],
    }
    print(f"Subscriber: {payload['subscriber_id']}  features: {len(payload['features'])}")

    response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    print(f"HTTP {response.status_code}")
    print(json.dumps(response.json(), indent=2))

    if response.status_code != 200:
        sys.exit("API validation FAILED")
    missing = EXPECTED_FIELDS - set(response.json())
    if missing:
        sys.exit(f"API validation FAILED: missing fields {sorted(missing)}")
    print("\nAPI validation passed.")


if __name__ == "__main__":
    main()
