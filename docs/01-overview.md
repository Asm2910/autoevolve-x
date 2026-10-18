# AutoEvolve-X: One-Page Overview

2026-10-18

AutoEvolve-X is a fraud-risk system in which a team of small agents reviews the data, evaluates and compares models, tests improvements and writes down what it learned, under strict rules. A language model may propose experiments. Deterministic Python runs them, measures them and decides. The held-out test set is sealed: agents never see it, and models are chosen on validation data only. The result is served through an API and monitored.

## The problem

Fraud is rare, so accuracy is meaningless and models are easy to fool. Letting an AI agent tune a model invites a worse failure: the agent quietly overfits to the data it is judged on. AutoEvolve-X is built so that cannot happen unnoticed.

## How it works

![System architecture](images/system-architecture.png)

A **mission planner** chooses steps from a depth (quick, standard or thorough). The **data swarm** checks the sources and features. The **evaluation swarm** finds where the model is weak. The **model arena** trains eight algorithms, and the **judge** picks the one to serve by business utility, not by the highest single score. The **improvement agent** tests changes and adopts one only if the paired bootstrap lower bound is above zero and the gain is at least 0.01 PR-AUC. **Reflection** writes grounded lessons to one memory, and a **research agent** answers questions about the project from that memory and the documents, with sources. A **harness** gives every agent a contract, a budget, a kill switch, an untrusted-text guard and a trace.

![Improvement loop](images/agent-loop.png)

## Results

Two datasets, never combined. Telecom is synthetic, with a hidden ground truth. IEEE-CIS is real e-commerce card fraud from Kaggle, used as validation on real data. It is not telecom data.

The model choice was made on validation only. The test table reports every model once and was not used to choose.

| Dataset | Test rows (positives) | Served model | Test PR-AUC [95% CI] | Logistic regression |
| --- | --- | --- | --- | --- |
| Telecom v2 (synthetic) | 46,291 (381) | xgboost | 0.292 [0.243, 0.356] | 0.213 [0.173, 0.265] |
| IEEE-CIS (real) | 60,000 (2,060) | lightgbm | 0.448 [0.425, 0.472] | 0.185 [0.173, 0.200] |

- On telecom the top models are tied within noise (random forest 0.300, extra trees 0.300, xgboost 0.292), so the judge served xgboost because it is the cheapest explainable model to run.
- On telecom the agent tried window changes and adopted none: every interval included zero or was below it. On IEEE-CIS it adopted deeper trees (+0.018 validation PR-AUC at depth 4, interval above zero).
- On IEEE-CIS the evaluation swarm found that fraud among payments with no identity record is caught at about 6.5%, against 37.8% overall.
- Ray, tried for parallel training, gave no speed-up on a 10-core laptop (0.93x with two workers, 0.79x with four).
- The older version-1 result (dataset v1, PR-AUC 0.272) is kept in `docs/v1_results`.

## What it does not claim

- The telecom data is synthetic, so its scores say the method works, not how it would do on real subscribers. IEEE-CIS is one real dataset, sampled to 300,000 rows.
- The planner chooses steps from the depth. It does not read the goal text.
- The prompt-injection filter is pattern-based. It is one layer, not a guarantee.
- The paired difference against logistic regression has not been recomputed for version 2.
- Probabilities are not calibrated, so read scores as rankings rather than true fraud rates.
- Drift detection recommends re-running the agent. Nothing retrains by itself.
- Alerts have no delivery channel, and the API has no authentication.
- The compose stack and the CatBoost-served Docker path were checked by hand only.

## Stack and what ships

Everything runs from one repository, `Asm2910/autoevolve-x`, and 291 automated tests pass.

| Layer | Tools |
| --- | --- |
| Modeling | XGBoost, LightGBM, CatBoost, scikit-learn, bootstrap confidence intervals |
| Agents | LangGraph, Claude through `langchain-anthropic`, rule-based fallback, deterministic swarms |
| Parallelism | Sequential by default, Ray optional |
| Memory | SQLite with FTS5, ChromaDB, reciprocal rank fusion |
| Retrieval | BM25 over the project's documents, reports and memory |
| Tracking | MLflow on SQLite, or a compose server |
| Serving | FastAPI in a non-root Docker image |
| Monitoring | Prometheus, Pushgateway, Grafana (API and agents dashboards), eight alert rules, PSI and KS drift checks |
| Review and control | Streamlit: ledger, mission control, ask page |
| Delivery | GitHub Actions: ruff, pytest, a quick mission, Docker build and health check |

## Where to go next

[Architecture](ARCHITECTURE.md) shows the diagrams, the [design document](02-design-document.md) explains the choices, the [technical document](03-technical-document.md) is the reference and runbook, the [data dictionary](04-data-dictionary.md) lists every column, and the [end-to-end guide](05-end-to-end-guide.md) tells the whole story in order. [PENDING.md](PENDING.md) lists what is still open.
