import json
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response

from api.schemas import PredictionRequest, PredictionResponse
from observability.logging import get_logger
from observability.metrics import (
    API_REQUEST_COUNT,
    API_REQUEST_LATENCY,
    PREDICTION_COUNT,
    render_metrics,
)

ARTIFACT_DIR = Path("models/artifacts")
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

logger = get_logger()

metadata = json.loads(METADATA_PATH.read_text())
model = joblib.load(ARTIFACT_DIR / metadata["artifact"])

MODEL_NAME = metadata["model_name"]
MODEL_VERSION = str(metadata["model_version"])
THRESHOLD = float(metadata["threshold"])
REQUIRED_FEATURES = list(metadata["features"])
NUMERIC_FEATURES = list(metadata["numeric_features"])

app = FastAPI(title="AutoEvolve-X Risk API", version=MODEL_VERSION)


def build_frame(features: dict) -> pd.DataFrame:
    """One-row frame with exactly the columns the model was trained on."""
    missing = [name for name in REQUIRED_FEATURES if name not in features]
    if missing:
        raise ValueError(f"missing {len(missing)} features, e.g. {missing[:5]}")

    frame = pd.DataFrame([{name: features[name] for name in REQUIRED_FEATURES}])
    frame[NUMERIC_FEATURES] = frame[NUMERIC_FEATURES].apply(pd.to_numeric, errors="raise")
    return frame.astype({name: "float64" for name in NUMERIC_FEATURES})


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        duration = time.perf_counter() - start
        path = request.url.path
        API_REQUEST_COUNT.labels(request.method, path, str(status_code)).inc()
        API_REQUEST_LATENCY.labels(request.method, path).observe(duration)
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": path,
                "status_code": status_code,
                "duration_seconds": round(duration, 4),
            },
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_name": MODEL_NAME, "model_version": MODEL_VERSION}


@app.get("/model/info")
def model_info() -> dict:
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "n_features": len(REQUIRED_FEATURES),
        "probability_calibrated": metadata.get("probability_calibrated", False),
        "metrics": metadata.get("metrics", {}),
        "test_ci95": metadata.get("test_ci95", {}),
        "training": metadata.get("training", {}),
    }


@app.get("/metrics")
def metrics() -> Response:
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    try:
        frame = build_frame(payload.features)
        probability = float(model.predict_proba(frame)[0, 1])
    except Exception as exc:  # missing or invalid features
        logger.warning(
            "prediction_failed",
            extra={"request_id": request.state.request_id, "error": str(exc)},
        )
        raise HTTPException(status_code=422, detail=f"Invalid features: {exc}") from exc

    prediction = int(probability >= THRESHOLD)
    PREDICTION_COUNT.labels(MODEL_NAME, str(prediction)).inc()
    logger.info(
        "prediction_made",
        extra={
            "request_id": request.state.request_id,
            "subscriber_id": payload.subscriber_id,
            "risk_probability": round(probability, 6),
            "risk_prediction": prediction,
        },
    )
    return PredictionResponse(
        subscriber_id=payload.subscriber_id,
        risk_probability=probability,
        risk_prediction=prediction,
        threshold=THRESHOLD,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        request_id=request.state.request_id,
    )
