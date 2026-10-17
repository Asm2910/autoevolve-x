# AutoEvolve-X: Technical Document

2026-10-17

This document records what was built, how each part works, and how to run and operate it. Repository: `Asm2910/autoevolve-x`. The one-page overview and the design document are the shorter companions to this one.

## Project history

The project grew in nine steps, each on its own git branch and merged into `main` with a merge commit.

| Step | Branch | What was built | What it proved |
| --- | --- | --- | --- |
| 1 | data-v2 | Synthetic hidden-fraud data, snapshot features, temporal split | The pipeline has a known ground truth |
| 2 | phase2-mlflow | MLflow tracking for training and ablations | Every run is logged and comparable |
| 3 | phase3-memory | SQLite memory with FTS5 and a ChromaDB index | Past work can be searched and reused |
| 4 | phase4-agent | LangGraph agent with a rule-based proposer | The loop runs end to end with deterministic decisions |
| 4b | phase4b-llm-proposer | Claude proposer with structured output and fallback | The model can propose without being trusted |
| 5 | phase5-final-test | Sealed-test confirmation, run once | Champion beats logistic regression by +0.061 PR-AUC |
| 6 | phase6-drift | PSI and KS drift checks, simulated drift | Drift is detected without labels |
| 7 | phase7-dashboard | Streamlit dashboard with an experiment ledger | Results can be reviewed visually |
| 8 | phase8-observability | Prometheus, Grafana, alerts, traffic generator, GitHub Actions CI | The live service is watched and the repo is checked on every change |
| 9 | phase9-docs | These documents and the README | The project can be understood in minutes |

Before step 1 the repository already held the training code, a FastAPI service and a Docker image. Those were extended rather than rewritten.

## Repository layout and module reference

```
autoevolve-x/
  agents/            LangGraph agent, proposers, experiment specs
  api/               FastAPI app and request schemas
  dashboard/         Streamlit app, theme, charts, data loading
  models/            Training code, explanations, model artifacts
  monitoring/        Drift checks (PSI, KS)
  observability/     Metrics, logging, Prometheus and Grafana config
  scripts/           Command-line entry points
  tools/             Final evaluation of the champion
  tests/             Unit and API tests
  data/processed/    Parquet splits and reports (git-ignored)
  .github/workflows/ CI
  .streamlit/        Dashboard theme settings
  Dockerfile, docker-compose.yml, requirements*.txt
```

| Module | Responsibility |
| --- | --- |
| `agents/llm_proposer.py` | Claude proposer with schema-checked output and fallback |
| `agents/spec.py` | The experiment specification the agent proposes |
| `tools/final_evaluation.py` | Fit champion and baseline, score test once, paired bootstrap |
| `monitoring/drift.py` | PSI, KS, drift summary, drift simulation |
| `api/app.py` | Endpoints, request metrics, score histogram |
| `observability/metrics.py` | Prometheus metric definitions |
| `dashboard/app.py` | Dashboard entry point; reads memory only |
| `scripts/run_agent.py` | Run the agent for N iterations |
| `scripts/confirm_champion.py` | Sealed final test, refuses a second run |
| `scripts/monitor_drift.py` | Drift check, real or simulated |
| `scripts/generate_traffic.py` | Send validation rows to the live API |

## Data pipeline and features

The data is a synthetic world in which some subscribers are fraudulent but the label is hidden behind noisy behaviour. Because the generator is ours, the true signal is known and the pipeline can be checked against it.

- **Snapshots.** Features are computed per subscriber per snapshot. Nine snapshots are split by time: 5 for training, 2 for validation and 2 for test.
- **Feature groups.** Customer, usage, recharge, network and risk. The dashboard colours features by these groups, and drift simulation scales one group at a time.
- **Files.** `train.parquet`, `validation.parquet`, `test.parquet` and `feature_manifest.json` live in `data/processed/`, which git ignores. The manifest lists numeric and categorical columns.
- **Positive rate.** About 0.9%, so PR-AUC is the headline metric.
- **Dataset identity.** A SHA-256 of the data is stored with every experiment, so memory knows which data a result came from.

The served model's metadata (`model_metadata.json`) lists the exact feature names, numeric columns, threshold and test confidence intervals the API needs.

## Modeling, evaluation and the final sealed test

Two model families are compared: gradient-boosted trees (XGBoost) and logistic regression as the honest baseline. Thresholds are chosen on validation data, never on test.

The final check lives in `tools/final_evaluation.py`:

1. Fit the champion and a logistic-regression baseline on the training snapshots.
2. Choose each model's threshold on validation.
3. Score both on the test snapshots, once.
4. Run a paired bootstrap on the same test rows. The verdict is `confirmed` if the lower bound of the difference is above zero.

`scripts/confirm_champion.py` wraps this. It refuses to run a second time for the same dataset hash, and it writes the result to memory, a lesson, an MLflow experiment and a markdown report.

| Result on test | PR-AUC | 95% CI |
| --- | --- | --- |
| Champion | 0.272 | 0.225 to 0.324 |
| Logistic regression | 0.211 | not reported |
| Paired difference | +0.061 | +0.030 to +0.094 |

With about 0.9% positives, 0.272 is roughly 30 times what a random scorer would reach.

## Experiment tracking with MLflow

MLflow uses a local SQLite backend. Each stage writes to its own experiment so results never mix.

| Experiment | Written by |
| --- | --- |
| `autoevolve-x` | Training runs, as a parent run with child runs |
| `autoevolve-x-ablation` | Ablation runs |
| `autoevolve-x-agent` | Agent experiments |
| `autoevolve-x-final` | The one-time sealed test |

MLflow stores the run details and charts. The agent's memory stores the decisions and the reasons for them. The two are linked by experiment name.

## Memory system internals

SQLite is the source of truth. ChromaDB holds a vector index that can be rebuilt from SQLite at any time.

| Part | Detail |
| --- | --- |
| Tables | `experiments` and `lessons`, plus an FTS5 virtual table for keyword search |
| Vector index | ChromaDB, with a hashing embedder or MiniLM chosen by the `--embedder` option |
| Hybrid search | Keyword and vector rankings merged by reciprocal rank fusion |
| Idempotency | `record_experiment` keys on kind, name, configuration hash and dataset hash, so repeats are no-ops |
| Kinds | `agent_experiment`, `final_test`, `drift_check`, plus rows ingested from training and ablation |
| Lessons | Short plain-language findings the agent recalls in later rounds |

Two rules keep memory trustworthy. Simulated drift checks are never recorded, so they cannot become lessons. The `final_test` kind doubles as the guard that blocks a second sealed-test run.

## Agent internals

![Agent round: 7 steps, 2 gates, 1 fallback](images/agent-flowchart.png)

The highlighted gate is plain Python. The dashed path runs only when the language model call fails. A round that adopts nothing is still recorded, so later rounds can learn from it.

- **Model call.** `ChatAnthropic` with no temperature set, using structured output against the `LLMProposal` schema. Hyperparameters are explicit optional fields, so a proposal cannot smuggle in unchecked settings.
- **Fallback.** A wrapper tries the model first and switches to the rule proposer on any failure, then continues the round as normal.
- **Novelty.** Compares the proposal's configuration hash with past experiments in memory.
- **Gate.** Adopt only if the bootstrap interval's lower bound is above zero and the mean gain is at least 0.01.
- **Record.** Writes the experiment idempotently to SQLite and the vector index, logs an MLflow run, and may add a lesson.
- **Stopping.** The model may end the session early. In the live run it stopped after ten experiments with none adopted.

## Drift monitoring: PSI and KS

Drift is measured on features, with no labels, because fraud labels arrive late. For each feature the monitor compares a reference window with a current window.

The population stability index sums, over bins, the gap between the current share a and the reference share e:

```latex
PSI = \sum_i (a_i - e_i) \ln\left(\frac{a_i}{e_i}\right)
```

- **Bins.** Quantile bins taken from the reference window. Missing values get their own bin. Categorical features use their categories.
- **KS test.** A Kolmogorov-Smirnov statistic is reported alongside PSI for numeric features.
- **Thresholds.** PSI of 0.10 is moderate and 0.25 is major. A retrain-share setting of 0.05 decides when the summary recommends re-running the agent.
- **Simulation.** `--simulate usage:1.6` scales every numeric feature in one group, to rehearse drift.

| Check | Outcome |
| --- | --- |
| Real data, reference against current | Stable |
| Simulated, usage group scaled by 1.6 | Drift: 36 features major, 17 moderate |

Real checks are recorded in memory as `drift_check` experiments with a lesson. Simulated ones are written only to `drift_simulated.csv`.

## API, Docker and observability

The API loads the champion artifact and its metadata at start-up, and refuses requests that miss features.

| Endpoint | Purpose |
| --- | --- |
| `POST /predict` | Score one subscriber: probability, flag, threshold, model version, request ID, top reasons |
| `GET /health` | Status, model name and version |
| `GET /model/info` | Threshold, feature count, metrics and test intervals |
| `GET /metrics` | Prometheus metrics |

**Metrics.** `api_requests_total` and `api_request_latency_seconds` (labelled by route pattern), `predictions_total`, `prediction_score` (histogram), `prediction_failures_total`, and the gauges `model_info` and `model_threshold`.

**Alerts** in `observability/alerts.yml`:

| Alert | Fires when | Severity |
| --- | --- | --- |
| ApiDown | The API is unreachable for 1 minute | critical |
| HighErrorRate | More than 5% of requests are 5xx for 5 minutes | critical |
| SlowPredictions | p95 of `/predict` is above 0.5 s for 5 minutes | warning |
| ScoreDistributionShift | Share of scores above 0.1 in 1 hour is over twice the 24-hour share, with more than 200 predictions | warning |
| RejectedPredictions | More than 20 rejected requests in 15 minutes | warning |

**Docker.** The image is `python:3.12-slim`, runs as a non-root user and copies only the API, observability code and model artifacts. Compose starts the API on port 8000, Prometheus on 9090 and Grafana on 3000.

**Grafana.** One provisioned dashboard with nine panels: requests per second, p95 latency, server errors, share flagged, traffic by endpoint, latency percentiles, score percentiles over time, flagged share and a score heatmap.

## Testing and CI

The last full run passed 101 tests, and `ruff check .` is clean. Tests cover each layer, and the configuration files are tested too.

| Area | What the tests check |
| --- | --- |
| API (`tests/test_api.py`) | Health, prediction validity, rejected input, metrics, bounded path labels |
| Agent (`test_agent.py`, 23 tests) | Loop behaviour, decision rule, novelty, fallback |
| LLM wiring (`test_llm_wire.py`) | Proposer schema and fallback; skipped if optional packages are missing |
| Final test (`test_final_evaluation.py`, 4 tests) | Verdict logic and the single-run guard |
| Drift (`test_drift.py`, 8 tests) | PSI, KS, missing-value bin, simulation |
| Dashboard (`test_dashboard.py`, 13 tests) | Data loading, charts, and a start-up test of the app |
| Monitoring config (`test_observability_config.py`) | Every dashboard and alert metric exists; Prometheus targets the right service and port; compose mounts exist |
| Memory and tracking | SQLite, search, idempotency, MLflow runs |

**GitHub Actions** runs on pushes to `main` and on pull requests:

1. **Lint and test:** install dev requirements, `ruff check .`, `pytest -q`.
2. **Build the API image:** validate the compose file, build the image, start it and wait for `/health`.

The second job starts only if the first passes, and a newer run cancels an older one on the same branch.

## How to run everything

Run these from the repository root with the virtual environment active.

```bash
# checks
python -m ruff check .
python -m pytest -q

# agent and evidence
python -m scripts.run_agent --iterations 3 --proposer llm
python -m scripts.confirm_champion          # sealed test, runs once per dataset
python -m scripts.monitor_drift             # real drift check
python -m scripts.monitor_drift --simulate usage:1.6

# dashboard (use python -m so the venv is used)
python -m streamlit run dashboard/app.py

# live service and monitoring
docker compose up -d --build
python -m scripts.generate_traffic --count 300 --rate 10
python -m scripts.generate_traffic --count 300 --rate 10 --shift usage:1.6
```

| Service | Address |
| --- | --- |
| API and interactive docs | http://localhost:8000 and http://localhost:8000/docs |
| Prometheus (targets, alerts) | http://localhost:9090 |
| Grafana | http://localhost:3000 (viewing is anonymous; editing uses `admin`) |

The language-model proposer reads its API key from the environment. Never put a key in a file or commit it.

## Appendix A: Model card

| Field | Detail |
| --- | --- |
| Model | The champion from the training pipeline; name, version and threshold are returned by `GET /model/info` |
| Task | Rank subscribers by fraud risk and flag those above a threshold |
| Training data | Synthetic hidden-fraud world, five training snapshots |
| Evaluation | PR-AUC with bootstrap intervals; threshold chosen on two validation snapshots; one sealed test on two further snapshots |
| Test result | PR-AUC 0.272 (95% CI 0.225 to 0.324), against 0.211 for logistic regression; paired difference +0.061 (+0.030 to +0.094) |
| Intended use | Demonstration of a verifiable model-improvement workflow; ranking for human review |
| Not for | Automatic denial of service, or use on real subscribers without retraining and revalidation on real data |
| Calibration | Scores are not calibrated probabilities; the flag `probability_calibrated` in the metadata reports this |
| Explanations | Each prediction returns up to five features that pushed the score up |
| Known limits | Synthetic data; rare positives make intervals wide; behaviour under real drift is untested |
| Fairness | No protected attributes were modelled or audited; a real deployment would need that review, and a person should review every flag |

## Appendix B: Operations runbook

| Symptom | Check | Action |
| --- | --- | --- |
| ApiDown alert, or `/health` fails | `docker compose ps`; `docker logs autoevolve-x-api` | Restart with `docker compose up -d`. If it exits at start-up, confirm `models/artifacts` holds the model file and `model_metadata.json` |
| HighErrorRate | Grafana panel Server errors; API logs filtered by request ID | Find the failing endpoint, fix or roll back to the previous image |
| SlowPredictions | Grafana latency percentiles | Check host load; the explanation step adds time, so look there first |
| ScoreDistributionShift | Grafana score percentiles and heatmap | Run `python -m scripts.monitor_drift`. If it reports drift, re-run the agent and plan a retrain |
| RejectedPredictions | Grafana rejected-requests line; log entries named `prediction_failed` | A client changed its features. Compare with the feature count from `/model/info` |
| Grafana panels empty | `http://localhost:9090/targets` shows the API as UP? | If DOWN, check the compose service name and port. If UP, send traffic with `generate_traffic` |
| `confirm_champion` refuses to run | Expected: the sealed test runs once per dataset | A new evaluation needs a new dataset, never a rerun on the same one |
| CI docker job fails | Container logs printed by the job | Confirm model artifacts are committed and the image starts locally |

No notifications are wired up yet, so alerts are visible only in Prometheus and Grafana.

## Appendix C: Documentation checklist

The documents below cover design, build, data and operation.

| Document | Status | Where |
| --- | --- | --- |
| One-page overview | Done | `docs/01-overview.md` |
| Design document | Done | `docs/02-design-document.md` |
| Technical document | Done | This document |
| Data dictionary | Done | `docs/04-data-dictionary.md` |
| End-to-end guide | Done | `docs/05-end-to-end-guide.md` |
| Model card | Done | Appendix A |
| Operations runbook | Done | Appendix B |
| README with quickstart | Done | `README.md` |
| Security note | Done | `SECURITY.md`, and `.env.example` |
| LICENSE | Done | `LICENSE` (MIT) |
| API reference | Partial | FastAPI serves interactive docs at `/docs`; no written guide |
| Dashboard screenshot | Done | `docs/images/dashboard-example.png`; replace with one from your own run |
| Grafana screenshot | Missing | Take it from your running stack and save as `docs/images/grafana-dashboard.png` |
| Change log | Not needed yet | The merge commits on `main` already record each phase |
