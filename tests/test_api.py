import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.app import THRESHOLD, app, model

TEST_DATA = Path("data/processed/test.parquet")

client = TestClient(app)


@pytest.fixture(scope="module")
def payload():
    if not TEST_DATA.exists():
        pytest.skip("test split not found")

    required = list(getattr(model, "feature_names_in_", []))
    row = pd.read_parquet(TEST_DATA).iloc[[0]]
    if not required or not set(required) <= set(row.columns):
        pytest.skip("model artifact was trained on a different feature set - retrain it")

    features = json.loads(row[required].to_json(orient="records"))[0]
    return {"subscriber_id": str(row["subscriber_id"].iloc[0]), "features": features}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info():
    body = client.get("/model/info").json()
    assert body["threshold"] == THRESHOLD
    assert "metrics" in body


def test_predict_returns_valid_probability(payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["risk_prediction"] == int(body["risk_probability"] >= THRESHOLD)
    assert body["subscriber_id"] == payload["subscriber_id"]
    assert body["request_id"]
    assert isinstance(body["top_reasons"], list)
    assert len(body["top_reasons"]) <= 5
    assert all(r["contribution"] > 0 for r in body["top_reasons"])


def test_predict_rejects_missing_features():
    response = client.post(
        "/predict", json={"subscriber_id": "X", "features": {"foo": 1}}
    )
    assert response.status_code == 422


def test_predict_rejects_missing_subscriber_id():
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 422


def test_metrics_endpoint_exposes_counters(payload):
    client.post("/predict", json=payload)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "predictions_total" in text
