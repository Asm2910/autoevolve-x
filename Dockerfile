FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# XGBoost needs the OpenMP runtime, which the slim image does not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# One source of truth for versions: use requirements.txt, but swap xgboost for its
# CPU-only build (same module and version, without ~300 MB of GPU libraries) and leave
# CatBoost out (~270 MB). CatBoost is installed below only if it is the model being served.
COPY requirements.txt .
RUN sed -e 's/^xgboost==/xgboost-cpu==/' -e '/^catboost==/d' requirements.txt > requirements.docker.txt \
    && pip install --upgrade pip \
    && pip install -r requirements.docker.txt

# Only what the API needs at runtime. The model files must exist locally:
# run `python -m scripts.train_models` before building.
COPY api ./api
COPY observability ./observability
COPY models/__init__.py models/explain.py ./models/
COPY arena/__init__.py arena/estimators.py ./arena/
COPY models/artifacts ./models/artifacts

# Install the library behind the served model if it is not already in the image.
# --no-deps skips CatBoost's plotting stack (matplotlib, plotly, graphviz), which inference never uses.
RUN served=$(python -c "import json; print(json.load(open('models/artifacts/model_metadata.json'))['model_name'])") \
    && echo "Served model: $served" \
    && if [ "$served" = "catboost" ]; then \
         pip install --no-deps "$(grep '^catboost==' requirements.txt)" six; \
       fi

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
