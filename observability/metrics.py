from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

API_REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "path", "status_code"],
)

API_REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "path"],
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total predictions made",
    ["model_name", "prediction"],
)

# Fraud is rare, so most scores are small: the buckets are dense near zero. A shift in this
# distribution is the live-traffic version of data drift, and it needs no labels to notice.
PREDICTION_SCORE = Histogram(
    "prediction_score",
    "Distribution of predicted fraud probabilities",
    ["model_name"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0),
)

PREDICTION_FAILURES = Counter(
    "prediction_failures_total",
    "Prediction requests rejected because the features were missing or invalid",
    ["reason"],
)

MODEL_INFO = Gauge(
    "model_info",
    "Which model is being served (the value is always 1)",
    ["model_name", "model_version"],
)

MODEL_THRESHOLD = Gauge("model_threshold", "Decision threshold of the served model", ["model_name"])


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST