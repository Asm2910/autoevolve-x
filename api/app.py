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
MODEL_PATH = ARTIFACT_DIR / "baseline_logistic_regression.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

logger = get_logger()

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text())

MODEL_NAME = metadata.get("model_name", "baseline_logistic_regression")
MODEL_VERSION = str(metadata.get("model_version", "v1"))
THRESHOLD = float(metadata.get("threshold", 0.90))

app = FastAPI(title="AutoEvolve-X Risk API", version=MODEL_VERSION)


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
        "metrics": metadata.get("metrics", {}),
    }


@app.get("/metrics")
def metrics() -> Response:
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    frame = pd.DataFrame([payload.features])
    try:
        probability = float(model.predict_proba(frame)[0, 1])
    except Exception as exc:  # missing/invalid features
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