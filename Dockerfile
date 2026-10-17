FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# XGBoost needs the OpenMP runtime, which the slim image does not include.
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# One source of truth for versions: use requirements.txt, but swap xgboost for its
# CPU-only build (same module and version, without ~300 MB of GPU libraries).
COPY requirements.txt .
RUN sed 's/^xgboost==/xgboost-cpu==/' requirements.txt > requirements.docker.txt \
    && pip install --upgrade pip \
    && pip install -r requirements.docker.txt

# Only what the API needs at runtime. The model files must exist locally:
# run `python -m scripts.train_models` before building.
COPY api ./api
COPY observability ./observability
COPY models/artifacts ./models/artifacts

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
