# AutoEvolve-X

A telecom fraud-risk system built to be **reproducible, honestly evaluated and observable**,
and designed to grow into a self-improving agent (experiments, memory, drift-triggered retraining).

> Status: data, features, models, API, Docker and monitoring hooks are done.
> Agents, MLflow tracking, memory and the dashboard are in progress (see Roadmap).

## What it does

Scores every active subscriber each month for the chance that fraud is confirmed in the next 30 days.

- **Synthetic data with a hidden ground truth.** The fraud state of each subscriber is decided first
  (`data_generation/world.py`) and events are generated *from* it, so labels are not a function of
  the features. Includes legitimate look-alikes (heavy callers, travellers, night workers), label noise,
  delayed confirmation and fraud-type drift over time.
- **Leak-free features.** Monthly snapshots; features use only events in `(T - window, T]`
  for 7/30/90-day windows plus trend features. Customers already confirmed before `T` are excluded.
- **Temporal evaluation.** First 5 snapshots train, next 2 choose thresholds, last 2 are scored once.
  Bootstrap 95% confidence intervals and paired model comparisons. Precision/recall at the top 1% of scores.
- **Models.** Logistic regression baseline and XGBoost. Probabilities are *not* calibrated
  (`probability_calibrated: false` in the model metadata).
- **Service.** FastAPI (`/health`, `/model/info`, `/predict`, `/metrics`) with JSON logs,
  request IDs and Prometheus metrics. Runs as a non-root user in Docker.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m scripts.run_pipeline          # generate -> validate -> features -> split -> train (~ minutes)
python -m pytest -q                      # tests
python -m uvicorn api.app:app --port 8000
python -m scripts.test_api               # sends one real test row
```

Docker (needs trained models in `models/artifacts/` first):

```bash
docker compose up -d --build
curl localhost:8000/health
```

Pipeline steps can be run alone: `scripts.generate_data`, `scripts.validate_data`,
`scripts.build_features`, `scripts.split_data`, `scripts.train_models`.
Configuration lives in `configs/data.yaml` and `configs/models.yaml`; everything is seeded.

## Layout

```
configs/               data and model settings
data_generation/       hidden world, behaviour profiles, event generators, snapshots, validation
feature_engineering/   windowed feature builder, validation
models/                training / evaluation helpers
scripts/               runnable pipeline steps
api/  observability/   FastAPI service, JSON logging, Prometheus metrics
tests/                 unit and API tests
```

Generated data (`data/`) and trained models (`models/artifacts/`) are not committed; the pipeline recreates them.
Each run writes `model_metadata.json` (dataset hash, git commit, metrics, thresholds) and `model_report.md`.

## Known limitations

- Data is synthetic. Absolute scores say nothing about real telecom fraud; the point is the method.
- Probabilities are uncalibrated; use scores for ranking and the validation-chosen threshold.
- Drift in the generator is mild, so retraining gains are small.

## Roadmap

1. MLflow experiment tracking
2. SHAP explanations and feature-group ablation
3. SQLite + ChromaDB memory of past experiments
4. LangGraph agents: propose experiments, run them, verify, decide
5. Drift detection and retraining trigger
6. Streamlit dashboard, Prometheus/Grafana, CI
