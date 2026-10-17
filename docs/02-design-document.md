# AutoEvolve-X: Design Document

2026-10-17

AutoEvolve-X lets a language model help improve a fraud-risk model without ever being able to cheat on the evaluation. This document explains the design choices and why they were made.

## Goals and non-goals

Goals:

- Improve a rare-event classifier through agent-proposed experiments, judged only on validation data.
- Make every decision reproducible and auditable from a single memory database.
- Serve the champion through an API that is monitored for traffic, speed, errors and score drift.
- Run the whole system from one repository, with CI that proves it still works.

Non-goals:

- No autonomous deployment: the agent never promotes a model to the API by itself.
- No real customer data, and no claim of production-grade accuracy.
- No large-scale AutoML search; the agent runs a handful of experiments per session.
- No data ingestion: the synthetic pipeline has no live feed to ingest.

## Design principles

The central rule is that the language model never decides anything that matters. It suggests, and deterministic code checks.

1. **Propose versus decide.** The model proposes experiments and explains them. Python runs them, measures them and applies a fixed adoption rule.
2. **A sealed test set.** The agent never reads the test snapshots. The champion is scored on test once per dataset, and a guard in memory refuses a second run.
3. **Uncertainty is part of the answer.** Every score carries a bootstrap confidence interval, and adoption needs the interval's lower bound above zero.
4. **Memory has one source of truth.** SQLite holds every experiment and lesson. The vector index is derived from it and can be rebuilt.
5. **Fail soft.** If the language model fails, a rule-based proposer takes over. If an explanation fails, scoring still succeeds.
6. **Rehearsals leave no trace.** Simulated drift checks are never written to memory, so they cannot teach the agent false lessons.

## System architecture

![System architecture: 11 components around one memory](images/system-architecture.png)

Data flows down the right-hand column from training to the served API and its monitoring. The left column is the learning side: the agent, the drift monitor and the dashboard all share one memory. The final test also records its result there.

## Data and evaluation design

The data is a synthetic world with hidden fraud, built so the answer is known and the pipeline can be tested honestly. Features are computed from snapshots of subscriber behaviour, and the split is by time, not at random.

| Split | Snapshots | Used for |
| --- | --- | --- |
| Train | 5 | Fitting models |
| Validation | 2 | Agent experiments, thresholds |
| Test | 2 | Final champion check, once |

A temporal split matters because behaviour changes over time, and a random split would let the model see the future. About 0.9% of rows are fraud, so the headline metric is PR-AUC rather than accuracy or ROC-AUC. Each score gets a bootstrap confidence interval, and comparisons between two models use a paired bootstrap on the same rows.

## Agent design

The agent is a LangGraph state machine with six nodes: recall, propose, novelty, run, decide, record. Each node does one job, so a failure is easy to locate.

- **Proposers.** Claude proposes through structured output validated against a schema, with explicit optional fields for each hyperparameter. A fallback wrapper switches to a rule-based proposer if the model call fails.
- **Novelty check.** A proposal that matches a past experiment by configuration is skipped, so budget is not spent on repeats.
- **Decision rule.** Adopt only if the confidence interval's lower bound is above zero and the mean gain is at least 0.01 PR-AUC.
- **Stopping.** The model may stop early when it judges that further ideas would be noise. In the live run it stopped after ten experiments, saying results had plateaued at this data size.

The decision rule is deliberately strict. With a 0.9% positive rate, small gains are usually noise, and a lenient rule would adopt luck.

## Memory design

Memory has two stores with different jobs. SQLite is the source of truth, and ChromaDB is a search index built from it.

- **Tables.** One for experiments and one for lessons, plus an FTS5 table for keyword search.
- **Hybrid search.** Keyword and vector results are merged by reciprocal rank fusion, so exact terms and similar meaning both count. Embeddings come from a hashing embedder or MiniLM.
- **Idempotent writes.** An experiment is identified by kind, name, configuration hash and dataset hash. Recording the same experiment twice changes nothing.
- **Kinds of record.** Agent experiments, the final test, drift checks, and rows ingested from training and ablation runs.

The same table backs the final-test guard: if a confirmation already exists for a dataset hash, the confirmation script refuses to run again.

## Serving and observability design

The champion is served by a small FastAPI service in a non-root Docker image. Prometheus scrapes it every 5 seconds and Grafana shows it, so a slow, failing or drifting service is visible within seconds.

- **Endpoints.** `/predict` scores one subscriber and returns the top reasons behind the score. `/health`, `/model/info` and `/metrics` cover status, model details and Prometheus.
- **Metrics.** Request count and latency, prediction counts, rejected requests, the served model and threshold, and a histogram of predicted scores with buckets dense near zero.
- **Bounded labels.** Metrics are labelled by route pattern, and unknown URLs share one label, so a scanner cannot create endless series.
- **Live drift signal.** The score histogram shows when scores move up, with no labels needed. An alert fires when the share of scores above 0.1 in the last hour is more than double the daily share.
- **Alerts.** Five rules: API down, high error rate, slow predictions, score shift and rejected requests.

## Key decisions and trade-offs

| Decision | Chosen | Alternative | Why |
| --- | --- | --- | --- |
| Who adopts a change | Fixed rule on the confidence interval | Let the model judge | A rule is auditable and cannot be talked into noise |
| Headline metric | PR-AUC | Accuracy or ROC-AUC | Only 0.9% of rows are fraud |
| Split | By time | Random | A random split leaks the future |
| Memory | SQLite plus a vector index | Vector database only | One source of truth, and the index can be rebuilt |
| Drift detection | PSI and KS on features | Track live accuracy | Fraud labels arrive late, so drift must be label-free |
| Dashboard charts | HTML and CSS only | A chart library | No JavaScript dependency, and full control of the look |
| Metric labels | Route pattern | Raw URL path | Keeps the number of series bounded |
| Alert delivery | Rules only, no Alertmanager | Add notifications | Kept in scope; listed as a gap |

## Risks and limitations

| Risk | Mitigation | Gap that remains |
| --- | --- | --- |
| The agent overfits validation | Strict adoption rule, sealed test | The same validation set is reused across experiments, with no correction for multiple comparisons |
| Synthetic data flatters the model | Known ground truth, honest caveats | Not yet tested on a real public dataset |
| Scores are not true probabilities | Threshold chosen on validation | No calibration step |
| The language model returns bad output | Schema validation, rule-based fallback | None known |
| Drift is detected but not acted on | Alert and a recommendation to re-run the agent | No ingestion or automatic retrain |
| Alerts have nowhere to go | Visible in Prometheus and Grafana | No Alertmanager, so no notifications |

## Roadmap and open questions

The remaining work is mostly about trust in the numbers, not new features.

- Run the pipeline on a real public fraud dataset to test the synthetic-data conclusions.
- Add probability calibration so scores can be read as rates.
- Correct for multiple comparisons when many experiments share one validation set.
- Add Alertmanager so alerts reach a person.
- Add a data-ingestion step so a drift alert can trigger a retrain.

Open question: should the agent be allowed to propose new features, not just model settings? That would widen what it can find, and would also widen the risk of leakage.
