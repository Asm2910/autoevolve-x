from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

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


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST