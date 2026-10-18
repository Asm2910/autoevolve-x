# AutoEvolve-X: Technical Document

2026-10-18

This document records what was built, how each part works, and how to run and operate it. Repository: `Asm2910/autoevolve-x`. The one-page overview and the design document are the shorter companions to this one. [ARCHITECTURE.md](ARCHITECTURE.md) has the diagrams.

## Project history

The project grew in 20 steps, each on its own git branch and merged into `main` with a merge commit. Steps 1 to 9 built version 1 (one dataset, one agent loop, one served model). Steps 10 to 20 built version 2.

| Step | Branch | What was built | What it proved |
| --- | --- | --- | --- |
| 1 | data-v2 | Synthetic hidden-fraud data, snapshot features, temporal split | The pipeline has a known ground truth |
| 2 | phase2-mlflow | MLflow tracking for training and ablations | Every run is logged and comparable |
| 3 | phase3-memory | SQLite memory with FTS5 and a ChromaDB index | Past work can be searched and reused |
| 4, 4b | phase4-agent, phase4b-llm-proposer | LangGraph improvement loop, rule-based and Claude proposers with a fallback | The loop runs end to end with deterministic decisions |
| 5 | phase5-final-test | Sealed-test confirmation, run once | The v1 champion beat logistic regression by +0.061 PR-AUC |
| 6 | phase6-drift | PSI and KS drift checks, simulated drift | Drift is detected without labels |
| 7 | phase7-dashboard | Streamlit dashboard with an experiment ledger | Results can be reviewed visually |
| 8 | phase8-observability | Prometheus, Grafana, alerts, traffic generator, GitHub Actions CI | The live service is watched |
| 9 | phase9-docs | Documents and README | The project can be understood in minutes |
| 10 | phase10-model-arena | Dataset v2 (fresh seed and sealed test), eight algorithms behind one registry, validation-only ranking | Models can be compared without touching the test set |
| 11 | phase11-ray-agent-platform | Ray executor, agent harness, budget and kill switch, guardrails, context builder, agent evaluations | Agents can be bounded and tested. Ray gave no speed-up here |
| 12 | phase12-evaluation-swarm | Performance, segment, error and drift agents | The model's weak spots are found by code |
| 13 | phase13-judge-utility | Utility judge picks the served model | Ties are decided by cost to run |
| 14 | phase14-data-swarm | Schema, quality, relationship and feature agents | The data is reviewed before any model |
| 15 | phase15-ieee-adapter | IEEE-CIS adapter and dataset switch | The system runs on real fraud data, kept separate from telecom |
| 16 | phase16-mission-graph | Mission planner, dispatch graph, reflection | One request runs the whole review, compare, improve and reflect cycle |
| 17 | phase17-research-agent | Local retrieval and a grounded research agent | The project can answer questions about itself, with sources |
| 18 | phase18-mission-ui | Mission control and ask pages in the dashboard | A mission can be started, watched and stopped from a browser |
| 19 | phase19-observability-v2 | Pushgateway, MLflow in compose, agents dashboard, agent alerts | Batch runs are visible in Grafana |
| 20, 21 | phase20-finalize, phase21-docs-v2 | README, architecture, CI smoke test, diagrams, these documents | The project is documented through version 2 |

Before step 1 the repository already held the training code, a FastAPI service and a Docker image. Those were extended rather than rewritten. [PROGRESS.md](PROGRESS.md) has a longer entry for each step, with the real numbers.

## Repository layout and module reference

```
autoevolve-x/
  agentkit/          Harness, budget, guardrails, tracing, context builder, data access, model doubles
  agents/            LangGraph improvement loop, proposers, decision rule, experiment spec
  api/               FastAPI app and request schemas
  arena/             Algorithm registry, runner, sequential and Ray executors
  dashboard/         Streamlit review page, mission control, ask page
  data_adapters/     IEEE-CIS adapter
  data_generation/   Synthetic world, event generators, snapshots, validation
  feature_engineering/ Windowed feature builder
  judge/             Utility judge
  memory/            SQLite store, ChromaDB index, embedders
  mission/           Planner, mission graph, reflection, report, service wiring
  models/            Training, explanations, MLflow tracking, model artifacts
  monitoring/        Drift checks (PSI, KS)
  observability/     API and agent metrics, Pushgateway push, alerts, Grafana provisioning
  research/          Corpus, retriever, research agent
  scripts/           Command-line entry points
  swarm/             Data swarm and evaluation swarm
  tools/             Experiment runner, final evaluation
  tests/             Unit, agent-behaviour and API tests
  data/              Parquet splits and reports (git-ignored)
  .github/workflows/ CI
  Dockerfile, docker-compose.yml, requirements*.txt, datasets.py
```

| Module | Responsibility |
| --- | --- |
| `datasets.py` | Which dataset is active (`AUTOEVOLVE_DATASET`), and the paths, config, model folder and memory file for it |
| `agentkit/harness.py` | `BaseAgent` and `AgentContract`: schema-checked input and output, allowed tools, timeout, retries |
| `agentkit/budget.py` | `Budget`: caps on calls, tokens, cost, time and experiments, and the kill-switch file |
| `agentkit/guardrails.py` | Sanitise and delimit untrusted text, injection patterns, cleaning model text |
| `agentkit/context.py` | Ranked, size-limited evidence for a prompt, and versioned prompts |
| `agentkit/tracing.py` | `Observer` and `TraceLog`: JSONL traces and metrics for every step |
| `agentkit/llm.py` | Test doubles (`ScriptedLLM`, `RecordingLLM`, `ReplayLLM`) and a token and cost tracker |
| `agentkit/data_access.py` | `load_split`: the only way agents read data; the test split is not reachable |
| `agents/graph.py`, `llm_proposer.py`, `proposers.py`, `decision.py`, `spec.py` | The improvement loop, its proposers, the adopt rule, the experiment specification |
| `arena/registry.py`, `runner.py`, `executor.py` | Eight algorithms, training and scoring, sequential and Ray execution |
| `judge/utility.py` | `JudgeAgent`: utility, paired bootstrap, explainability rule, tie-break |
| `swarm/data.py`, `swarm/evaluation.py` | The two swarms |
| `mission/plan.py`, `graph.py`, `reflection.py`, `services.py`, `report.py` | Planner, dispatch graph, reflection, real service wiring, report |
| `research/corpus.py`, `retriever.py`, `agent.py` | Passages, BM25 retrieval, the grounded research agent |
| `data_adapters/ieee_cis.py` | Read, join, build features, cap levels, split by time, manifest |
| `tools/final_evaluation.py` | Fit champion and baseline, score test once, paired bootstrap |
| `monitoring/drift.py` | PSI, KS, drift summary, drift simulation |
| `observability/agent_metrics.py`, `push.py` | Agent metrics registry and the Pushgateway push |
| `dashboard/app.py`, `control.py`, `pages/` | Review page, mission control logic and pages |
| `api/app.py` | Endpoints, request metrics, score histogram |

## Data pipeline and features

The telecom data is a synthetic world in which some subscribers are fraudulent but the label is hidden behind noisy behaviour. Because the generator is ours, the true signal is known and the pipeline can be checked against it.

- **Snapshots.** Features are computed per subscriber per snapshot. Nine snapshots are split by time: 5 for training, 2 for validation and 2 for test.
- **Feature groups.** Customer, usage, recharge, network and risk. The dashboard colours features by these groups, and drift simulation scales one group at a time.
- **Files.** `train.parquet`, `validation.parquet`, `test.parquet` and `feature_manifest.json` live in `data/processed/`, which git ignores. The manifest lists numeric and categorical columns.
- **Dataset v2.** Seed 20261018. 109,342 train rows, 46,589 validation rows (484 positives) and 46,291 test rows (381 positives). 254 features, 7 categorical.
- **Dataset identity.** A SHA-256 of the data is stored with every experiment, so memory knows which data a result came from.

The served model's metadata (`model_metadata.json`) lists the exact feature names, numeric columns, threshold, selection rule and test confidence intervals the API needs.

## IEEE-CIS adapter

`data_adapters/ieee_cis.py` turns the Kaggle files `train_transaction.csv` and `train_identity.csv` into the same shape the rest of the project uses. Prepare it with `python -m scripts.prepare_ieee --source ~/Downloads/ieee-fraud-detection.zip` (default: 300,000 payments drawn at random from the 590,540 in the file, time order kept; card history is computed on all payments before sampling).

- **Split.** About 180,000 train, 60,000 validation and 60,000 test rows, split by time: the earliest 60%, the next 20% and the last 20%. Rows with the same time stay together.
- **Features.** 440 in 11 groups (31 categorical): transaction, card, address, email, counts, timedeltas, match, vesta, identity, device and history. See the [data dictionary](04-data-dictionary.md).
- **History.** Built per card (the pair `card1` and `addr1`) from earlier payments only: payments so far, amount against the card's own average, seconds since the previous payment.
- **Level capping.** Categorical columns keep the 25 most common training values. Capping uses training rows only.
- **Separate everything.** With `AUTOEVOLVE_DATASET=ieee` the processed folder, raw folder, config (`configs/ieee.yaml`), model folder, memory database and vector index all switch.
- **Honesty.** This is real e-commerce card fraud, not telecom. The two datasets are never combined, and results are reported side by side.

## Modeling, evaluation and the final sealed test

Eight algorithms are compared in the arena: logistic regression, random forest, extra trees, XGBoost, LightGBM, histogram gradient boosting, CatBoost and isolation forest. Thresholds are chosen on validation data, never on test.

The final check lives in `tools/final_evaluation.py`:

1. Fit the champion and a logistic-regression baseline on the training snapshots.
2. Choose each model's threshold on validation.
3. Score both on the test snapshots, once.
4. Run a paired bootstrap on the same test rows. The verdict is `confirmed` if the lower bound of the difference is above zero.

`scripts/confirm_champion.py` wraps this. It refuses to run a second time for the same dataset hash, and it writes the result to memory, a lesson, an MLflow experiment and a markdown report.

| Result on test (dataset v1) | PR-AUC | 95% CI |
| --- | --- | --- |
| Champion | 0.272 | 0.225 to 0.324 |
| Logistic regression | 0.211 | not reported |
| Paired difference | +0.061 | +0.030 to +0.094 |

With about 0.9% positives, 0.272 is roughly 30 times what a random scorer would reach.

**Dataset v2 (current, seed 20261018, fresh sealed test).** The model choice was made on validation only; the test table below reports every model once and was not used to choose. Telecom: 381 positives in 46,291 test rows. IEEE-CIS (real e-commerce fraud, a separate validation, not telecom data): 2,060 positives in 60,000 test rows.

| Dataset | Served model | Test PR-AUC [95% CI] | Logistic regression |
| --- | --- | --- | --- |
| Telecom v2 | xgboost | 0.292 [0.243, 0.356] | 0.213 [0.173, 0.265] |
| IEEE-CIS | lightgbm | 0.448 [0.425, 0.472] | 0.185 [0.173, 0.200] |

On telecom v2 the top boosted and bagged models are statistically tied (random forest 0.300, extra trees 0.300, xgboost 0.292), which is why the judge chose by cost to run rather than by the highest point estimate. The paired difference against logistic regression has not been recomputed for v2.

## Model arena, executors and the judge

**Arena.** `scripts/run_arena.py` trains every algorithm on the same data and ranks them on the validation snapshots. Models whose paired interval overlaps the leader are reported as "tied with best". It writes `arena_report.md`, `arena_validation.json` and logs to MLflow (`autoevolve-x-arena`).

Validation PR-AUC on telecom v2: CatBoost 0.303, XGBoost 0.297, random forest 0.293, histogram gradient boosting 0.292, LightGBM 0.291, extra trees 0.288, logistic regression 0.256, isolation forest 0.128. On IEEE-CIS: LightGBM 0.558, XGBoost 0.548, histogram gradient boosting 0.546, CatBoost 0.531, random forest 0.522, logistic regression 0.466.

**Executors.** `make_executor(workers)` returns a sequential executor for one worker and a Ray executor for more. The Ray executor runs each model as a task with one CPU, retries a failed task once, and enforces a per-model timeout. A model that times out or fails is reported as failed and the others continue.

**Ray benchmark.** `scripts/benchmark_ray.py` times the arena at 1, 2 and 4 workers (7 models, 10 CPU cores).

| Workers | Executor | Wall clock (s) | Speed-up |
| --- | --- | --- | --- |
| 1 | sequential | 104.6 | 1.00x |
| 2 | ray | 112.2 | 0.93x |
| 4 | ray | 132.7 | 0.79x |

Results are identical across executors (maximum PR-AUC difference 0.0000). Ray did not help, because these models are small and process start-up and data transfer cost more than they save.

**Judge.** `judge/utility.py` picks the served model, and `scripts/train_models.py` records the verdict in `judge_report.md`, `judge_verdict.json` and the model metadata.

| Setting | Value |
| --- | --- |
| Fraud value (per case caught) | 400 |
| Review cost (per reviewed account) | 6 |
| False-alarm cost (extra, per false alarm) | 3 |
| Review depth | The share of accounts analysts can review (`review_fraction`: 1% for telecom, 2% for IEEE-CIS) |
| Explainable set | xgboost, lightgbm, catboost (`require_explanations: true`) |
| Tie rule | Gap interval includes zero |
| Tie-break | Cheapest to retrain, then smallest, unless more than 10% below the best utility |
| `training.selection` | `utility` (default); `pr_auc` restores the old rule |

On telecom v2 the judge served XGBoost (utility 16,054 per 10,000 accounts). CatBoost (15,968) and LightGBM (15,708) were tied with it. XGBoost trains in about 2 seconds and is 0.4 MB. Four models were excluded because they cannot explain a score. On IEEE-CIS the judge served LightGBM.

## The two swarms

Both swarms are deterministic agents on the shared harness. Each agent returns findings with a severity (info, warning, critical), evidence and an optional hypothesis the improvement agent may test. A swarm returns a verdict (healthy, watch or attention) and writes a markdown report and a findings file the improvement agent can read.

**Data swarm** (`swarm/data.py`, `scripts/run_data_review.py`). Works on raw tables and the training features. It cannot reach the test split.

| Agent | Checks |
| --- | --- |
| Schema | Expected columns, types and categories |
| Quality | Missing values, duplicate ids, constant columns, impossible values, odd timestamps, many near-empty columns |
| Relationship | Do sources join cleanly: orphans, coverage, hot keys, history before each snapshot |
| Feature | Per-feature signal (AUC or Cramér's V) with Benjamini-Hochberg false discovery control and an effect-size floor of 0.03. Group ranking, window ranking, and a leakage check at AUC 0.97 or more |

**Evaluation swarm** (`swarm/evaluation.py`, `scripts/run_evaluation.py`). Works on the current champion and validation data.

| Agent | Checks |
| --- | --- |
| Performance | How good the model is overall, with an interval, and whether it is steady over time |
| Segment | Segments weaker than overall, with Bonferroni-corrected bootstrap. Shows the top shortfalls plus a summary, and merges identical groups |
| Error | How the frauds it misses differ from the ones it catches |
| Drift | Whether the data being scored has stopped looking like the data the model learned from (PSI and KS on features) |

On the IEEE data the segment agent found that fraud among payments with no identity record is caught at about 6.5%, against 37.8% overall.

## Agent platform: harness, budget and guardrails

`agentkit/` is the common base for every agent.

| Part | Detail |
| --- | --- |
| Contract | `AgentContract(name, responsibility, kind, input_model, output_model, allowed_tools, timeout_seconds, max_retries)`. Input and output are pydantic models. A tool not in `allowed_tools` cannot be called |
| Budget | `Budget` caps language-model calls (60), input tokens (400,000), output tokens (60,000), cost (2.0 US dollars), seconds (3,600) and experiments (15). Values are in `agent_platform` in the config. `check()` raises `BudgetExceeded` |
| Kill switch | `memory/STOP`. If the file exists, `check()` raises `KillSwitchTripped` |
| Context | `ContextBuilder` ranks items and fits them in a token budget (6,000 by default). `versioned_prompt` gives each prompt a name and version |
| Untrusted text | `sanitize_untrusted` strips control characters, caps length (300 characters per item), and replaces text matching 11 injection patterns with a placeholder. `wrap_untrusted` delimits items in `<untrusted_data>` tags |
| Observer | Every step is traced to JSONL (`memory/traces/`) and counted in Prometheus metrics |
| Test doubles | `ScriptedLLM`, `RecordingLLM` and `ReplayLLM` let tests run the model paths without a key |

**Agent metrics** (own registry, `observability/agent_metrics.py`): `agent_execution_total`, `agent_execution_duration_seconds`, `agent_failures_total`, `llm_requests_total`, `llm_tokens_total`, `llm_cost_usd_total`, `experiments_total`, `evaluation_findings_total`, `agent_best_validation_pr_auc`, `agent_run_last_timestamp_seconds`, `agent_run_duration_seconds`, `mission_rounds_completed` and `mission_halted`.

The injection filter is pattern-based and is one layer of defence. Structured, schema-checked outputs and fixed decision rules are the stronger protection.

## Mission graph

`scripts/run_mission.py` runs a mission. The planner, dispatch graph and reflection are in `mission/`.

```bash
python -m scripts.run_mission --depth quick
AUTOEVOLVE_DATASET=ieee python -m scripts.run_mission --goal "Improve fraud detection" --depth standard --workers 4
```

| Depth | Improvement experiments per round | Rounds | Model arena |
| --- | --- | --- | --- |
| quick | 2 | 1 | no |
| standard | 4 | 2 | yes |
| thorough | 6 | 3 | yes |

`--rounds` and `--iterations` override the depth. The planner is rule-based and does not read the goal text.

- **Dispatch.** Runs the pending steps from the plan. A failed step is recorded and removed from the queue. A budget overrun or the kill switch stops the mission, which then reflects.
- **Re-planning.** After the data review and the evaluation the plan is revised and swarm ideas are passed to the improvement step.
- **Improvement.** Runs the experiments through the same runner and adopt rule as the standalone agent. The report shows the first round's starting score and the last round's ending score.
- **Reflection.** Writes lessons grounded in the results and decides whether to run another round. It does so only if the last round adopted a change and rounds remain.
- **Output.** `mission_report.md` and `mission_report.json` next to the processed data, and a JSONL trace.

## Research agent

`scripts/ask.py` and `research/`.

```bash
python -m scripts.ask "Why is xgboost the served model?"
python -m scripts.ask "What did the evaluation find about missed fraud?" --llm
```

- **Corpus.** README, SECURITY, `docs/*.md`, the processed report files, lessons and recent experiments from memory. Documents are cut at headings and paragraphs into passages of up to 900 characters, each with its source.
- **Retrieval.** BM25 over stemmed words with stop words removed. `--embedder hashing` or `minilm` adds embeddings by reciprocal rank fusion.
- **Without `--llm`.** The best three passages, each with its source.
- **With `--llm`.** The model answers from numbered passages and cites them. Code removes citations that do not name a shown passage, withholds an answer with no valid citation, and lists any number that does not appear in the cited passages as ungrounded. If nothing is retrieved or the model says the passages are insufficient, the answer is "I could not find this in the project's records."
- **Safety.** The question and every passage are sanitised and delimited as untrusted text. The agent obeys the same budget and kill switch.

## Experiment tracking with MLflow

MLflow uses a local SQLite backend (`mlflow.db`), or a server if `MLFLOW_TRACKING_URI` is set. Each stage writes to its own experiment so results never mix.

| Experiment | Written by |
| --- | --- |
| `autoevolve-x` | Training runs, as a parent run with child runs |
| `autoevolve-x-ablation` | Ablation runs |
| `autoevolve-x-arena` | The model arena |
| `autoevolve-x-agent` | Agent experiments |
| `autoevolve-x-final` | The one-time sealed test |

MLflow stores the run details and charts. The agent's memory stores the decisions and the reasons for them. The two are linked by experiment name. Docker compose can start an MLflow server on port 5001 (`export MLFLOW_TRACKING_URI=http://localhost:5001`). Port 5000 is avoided because macOS uses it for AirPlay. The client still writes artifacts to the local `mlartifacts/` folder.

## Memory system internals

SQLite is the source of truth. ChromaDB holds a vector index that can be rebuilt from SQLite at any time. Each dataset has its own database (`autoevolve.db`, `autoevolve_ieee.db`) and index.

| Part | Detail |
| --- | --- |
| Tables | `experiments` and `lessons`, plus an FTS5 virtual table for keyword search |
| Vector index | ChromaDB, with a hashing embedder or MiniLM chosen by the `--embedder` option |
| Hybrid search | Keyword and vector rankings merged by reciprocal rank fusion |
| Idempotency | `record_experiment` keys on kind, name, configuration hash and dataset hash, so repeats are no-ops |
| Kinds | `agent_experiment`, `final_test`, `drift_check`, plus rows ingested from training and ablation |
| Lessons | Short plain-language findings the agents recall in later rounds |

Two rules keep memory trustworthy. Simulated drift checks are never recorded, so they cannot become lessons. The `final_test` kind doubles as the guard that blocks a second sealed-test run.

## Improvement agent internals

![Agent round: 7 steps, 2 gates, 1 fallback](images/agent-flowchart.png)

The highlighted gate is plain Python. The dashed path runs only when the language model call fails. A round that adopts nothing is still recorded, so later rounds can learn from it. `scripts/run_agent.py` runs it on its own; `--use-findings` lets the proposer read the swarm findings as delimited, sanitised data.

- **Model call.** `ChatAnthropic` with no temperature set, using structured output against the `LLMProposal` schema. Hyperparameters are explicit optional fields, so a proposal cannot smuggle in unchecked settings.
- **Fallback.** A wrapper tries the model first and switches to the rule proposer on any failure, then continues the round as normal.
- **Novelty.** Compares the proposal's configuration hash with past experiments in memory.
- **Gate.** Adopt only if the bootstrap interval's lower bound is above zero and the mean gain is at least 0.01.
- **Record.** Writes the experiment idempotently to SQLite and the vector index, logs an MLflow run, and may add a lesson.
- **Budget.** Every step charges the budget and checks the kill switch.

Real results. Telecom v2: window changes were rejected (for example a 30-day change of -0.014, interval -0.030 to +0.002). IEEE-CIS: deeper trees were adopted (depth 4 gave +0.018 validation PR-AUC, depth 5 gave +0.014).

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

**API metrics.** `api_requests_total` and `api_request_latency_seconds` (labelled by route pattern), `predictions_total`, `prediction_score` (histogram), `prediction_failures_total`, and the gauges `model_info` and `model_threshold`.

![Monitoring flow](images/monitoring-flow.png)

**Pushgateway.** Missions and agent scripts finish before Prometheus would scrape them, so `run_mission`, `run_agent`, `run_data_review` and `run_evaluation` push their final metrics when they end (`observability/push.py`). It is off unless `PUSHGATEWAY_URL` or `agent_platform.pushgateway_url` is set, and a failure prints one line and never fails the run. Each (dataset, script) pair is its own group, so the dashboard shows the last run of each script, not a history. History lives in the JSONL traces and the experiment memory.

**Alerts** in `observability/alerts.yml`:

| Alert | Fires when | Severity |
| --- | --- | --- |
| ApiDown | The API is unreachable for 1 minute | critical |
| HighErrorRate | More than 5% of requests are 5xx for 5 minutes | critical |
| SlowPredictions | p95 of `/predict` is above 0.5 s for 5 minutes | warning |
| ScoreDistributionShift | Share of scores above 0.1 in 1 hour is over twice the 24-hour share, with more than 200 predictions | warning |
| RejectedPredictions | More than 20 rejected requests in 15 minutes | warning |
| AgentStepsFailed | The pushed metrics show failed agent steps | warning |
| LlmSpendHigh | Pushed language-model spend is above 1.50 US dollars | warning |
| AgentRunsStale | No agent script has reported for a week | warning |

Because pushed values are the last run, the two agent alerts stay on until the next run replaces them.

**Docker.** The API image is `python:3.12-slim`, runs as a non-root user and copies only the API, observability code and model artifacts. Compose starts the API (8000), Prometheus (9090), Grafana (3000), the Pushgateway (9091) and MLflow (5001). After switching git branches or unzipping over the repository, run `docker compose up -d --force-recreate`, because a container keeps following the folders as they were when it started.

**Grafana.** Two provisioned dashboards. *AutoEvolve-X API* has nine panels: requests per second, p95 latency, server errors, share flagged, traffic by endpoint, latency percentiles, score percentiles over time, flagged share and a score heatmap. *AutoEvolve-X agents* has twelve: time since the last run, champion validation PR-AUC by dataset, language-model spend and tokens, experiments by verdict, failed steps by agent, findings by severity, slowest 5% of steps by agent, language-model calls by agent, rounds in the last mission, whether it stopped early, and run time.

## Control panel

`python -m streamlit run dashboard/app.py` shows the review page, with two more pages in the sidebar.

- **Mission control** (`dashboard/control.py`, `dashboard/pages/1_Mission_control.py`). Enter a goal, pick the dataset and depth, preview the plan, start the mission, watch the live log and stop it. The mission runs as a separate process (`scripts.run_mission`) with its output in `memory/runs/`, so closing the tab does not stop it. Stopping creates the kill-switch file. Only one mission runs at a time, and a stale stop file is cleared before a new run. The page validates input through the planner before it starts anything.
- **Ask the project** (`dashboard/pages/2_Ask_the_project.py`). The research agent without a model: closest passages with sources.

## Testing and CI

The last full run passed 291 tests, and `ruff check .` is clean. Tests cover each layer, and the configuration files are tested too.

| Area | Tests | What they check |
| --- | --- | --- |
| API (`tests/test_api.py`) | 9 | Health, prediction validity, rejected input, metrics, bounded path labels |
| Data (`tests/test_data.py`, `test_snapshot_features.py`) | 12 | Generator and feature invariants |
| Improvement agent (`test_agent.py`) | 23 | Loop behaviour, decision rule, novelty, fallback |
| Agent platform (`test_agentkit.py`) | 45 | Contracts, budget, kill switch, guardrails, context, tracing, metrics |
| Agent behaviour (`tests/agent_evals/`) | 16 | Injection resistance, sealed-data scan, budget stops, behaviour of the whole loop |
| LLM wiring (`test_llm_wire.py`) | 1 | Proposer schema and fallback; skipped if optional packages are missing |
| Arena and executors (`test_arena.py`, `test_executor.py`) | 32 | Registry, ranking, sequential and Ray executors (Ray tests skip if it is not installed) |
| Judge (`test_judge.py`) | 11 | Utility, ties, exclusions, tie-break |
| Swarms (`test_eval_swarm.py`, `test_data_swarm.py`) | 32 | Each agent, corrections, deduplication, findings handoff |
| IEEE adapter (`test_ieee_adapter.py`) | 9 | Join, causal history, level capping, temporal split, manifest |
| Mission (`test_mission.py`) | 14 | Planner depths, dispatch, re-plan, failure containment, budget stop, report |
| Research (`test_research.py`) | 11 | Chunking, ranking, grounded answers, citation and number checks, injection |
| Control panel (`test_control.py`, `test_dashboard.py`) | 18 | Start, watch and stop a run, stale stop file, data loading, charts |
| Final test, drift, explain, memory, tracking, training | 41 | The v1 layers |
| Monitoring config (`test_observability_config.py`, `test_push.py`) | 17 | Every dashboard and alert metric exists, Prometheus targets, compose mounts, a real push to a stand-in gateway, push never raises |

Agent tests never need an API key: language-model paths run against scripted doubles.

**GitHub Actions** runs on pushes to `main` and on pull requests:

1. **Lint and test:** install dev requirements, `ruff check .`, build the data and train the fast candidates with `scripts.run_pipeline` (data and models are git-ignored, about two minutes), `pytest -q`, then run a quick mission and a research question on the freshly built data (rule-based, no key), and keep the trained model as a build artifact.
2. **Build the API image:** fetch that model, validate the compose file, build the image, start it and wait for `/health`.

The second job starts only if the first passes, and a newer run cancels an older one on the same branch.

## How to run everything

Run these from the repository root with the virtual environment active.

```bash
# checks
python -m ruff check .
python -m pytest -q

# data, models, judge
python -m scripts.run_pipeline              # generate, validate, features, split, train, judge
python -m scripts.run_arena --workers 1     # eight algorithms on validation
python -m scripts.benchmark_ray             # 1, 2 and 4 workers

# reviews and agents
python -m scripts.run_data_review
python -m scripts.run_evaluation
python -m scripts.run_agent --iterations 4 --use-findings
python -m scripts.run_agent --iterations 3 --proposer llm      # needs ANTHROPIC_API_KEY
python -m scripts.run_mission --depth standard
python -m scripts.ask "Why is xgboost the served model?"

# real data
python -m scripts.prepare_ieee --source ~/Downloads/ieee-fraud-detection.zip
AUTOEVOLVE_DATASET=ieee python -m scripts.run_arena
AUTOEVOLVE_DATASET=ieee python -m scripts.run_mission --depth standard

# evidence
python -m scripts.confirm_champion          # sealed test, runs once per dataset
python -m scripts.monitor_drift             # real drift check
python -m scripts.monitor_drift --simulate usage:1.6

# dashboard (use python -m so the venv is used)
python -m streamlit run dashboard/app.py

# live service and monitoring
docker compose up -d --build
export PUSHGATEWAY_URL=http://localhost:9091
python -m scripts.generate_traffic --count 300 --rate 10
python -m scripts.generate_traffic --count 300 --rate 10 --shift usage:1.6
touch memory/STOP                           # from another terminal: stop a running agent
```

| Service | Address |
| --- | --- |
| API and interactive docs | http://localhost:8000 and http://localhost:8000/docs |
| Prometheus (targets, alerts) | http://localhost:9090 |
| Grafana | http://localhost:3000 (viewing is anonymous; editing uses `admin`) |
| Pushgateway | http://localhost:9091 |
| MLflow | http://localhost:5001 |

The language-model proposer and the research agent's `--llm` mode read their API key from the environment. Never put a key in a file or commit it.

## Appendix A: Model card

| Field | Detail |
| --- | --- |
| Model | The served model chosen by the judge; name, version and threshold are returned by `GET /model/info` |
| Task | Rank subscribers (or payments) by fraud risk and flag those above a threshold |
| Training data | Telecom: synthetic hidden-fraud world, five training snapshots. IEEE-CIS: 180,000 real e-commerce payments, earliest in time |
| Evaluation | PR-AUC with bootstrap intervals; threshold chosen on validation; one sealed test per dataset on the most recent data |
| Selection | Judge: business utility per 10,000 accounts, explainable models only, ties decided by cost to run |
| Test result | Telecom v2: xgboost PR-AUC 0.292 (95% CI 0.243 to 0.356). IEEE-CIS: lightgbm 0.448 (0.425 to 0.472). Dataset v1: 0.272 against 0.211 for logistic regression |
| Intended use | Demonstration of a verifiable model-improvement workflow; ranking for human review |
| Not for | Automatic denial of service, or use on real subscribers without retraining and revalidation on real data |
| Calibration | Scores are not calibrated probabilities; the flag `probability_calibrated` in the metadata reports this |
| Explanations | Each prediction returns up to five features that pushed the score up |
| Known limits | Telecom data is synthetic; IEEE-CIS is one e-commerce dataset; rare positives make intervals wide; behaviour under real drift is untested |
| Fairness | No protected attributes were modelled or audited; a real deployment would need that review, and a person should review every flag |

## Appendix B: Operations runbook

| Symptom | Check | Action |
| --- | --- | --- |
| ApiDown alert, or `/health` fails | `docker compose ps`; `docker logs autoevolve-x-api` | Restart with `docker compose up -d`. If it exits at start-up, confirm `models/artifacts` holds the model file and `model_metadata.json` |
| HighErrorRate | Grafana panel Server errors; API logs filtered by request ID | Find the failing endpoint, fix or roll back to the previous image |
| SlowPredictions | Grafana latency percentiles | Check host load; the explanation step adds time, so look there first |
| ScoreDistributionShift | Grafana score percentiles and heatmap | Run `python -m scripts.monitor_drift`. If it reports drift, re-run the agent and plan a retrain |
| RejectedPredictions | Grafana rejected-requests line; log entries named `prediction_failed` | A client changed its features. Compare with the feature count from `/model/info` |
| AgentStepsFailed | The agents dashboard, "Failed steps by agent"; the latest `memory/traces/*.jsonl` | Read the error in the trace. The mission carried on, so check what was skipped. The alert clears on the next run |
| LlmSpendHigh | The agents dashboard, spend and tokens | Lower `max_cost_usd` or `max_llm_calls` in `agent_platform`, or use the rule-based proposer |
| AgentRunsStale | "Time since the last run" panel | Run a mission, or ignore it if nothing was meant to run |
| A mission must stop now | `touch memory/STOP`, or the Stop button in mission control | The mission ends at its next step and still writes its reflection. Delete the file before the next run (mission control does this for you) |
| Grafana panels empty | `http://localhost:9090/targets` shows the API as UP? | If DOWN, check the compose service name and port. If UP, send traffic with `generate_traffic` |
| Agents dashboard empty | `http://localhost:9091` lists a group? Is `PUSHGATEWAY_URL` set in the shell that ran the mission? | Export the variable and run a mission. The script prints "metrics pushed" when it works |
| Grafana says "Failed to load home dashboard", or `ls /etc/grafana/dashboards` in the container is empty | `docker compose exec grafana ls /etc/grafana/dashboards` | A container keeps following the folders as they were when it started, so a branch switch can leave it with an empty folder. Run `docker compose up -d --force-recreate prometheus grafana` |
| `docker compose up` says port 5000 is in use | macOS AirPlay Receiver | MLflow is published on 5001 for this reason |
| `confirm_champion` refuses to run | Expected: the sealed test runs once per dataset | A new evaluation needs a new dataset, never a rerun on the same one |
| `pytest` reports `No module named ...` | `which python` | The virtual environment is not active. Run `source .venv/bin/activate` and use `python -m pytest` |
| CI docker job fails | Container logs printed by the job | Model artifacts are not in git. Confirm the test job uploaded `model-artifacts`, and that the image starts locally after `python -m scripts.run_pipeline` |

No notifications are wired up yet, so alerts are visible only in Prometheus and Grafana.

## Appendix C: Documentation checklist

The documents below cover design, build, data and operation.

| Document | Status | Where |
| --- | --- | --- |
| One-page overview | Done | `docs/01-overview.md` |
| Design document | Done | `docs/02-design-document.md` |
| Technical document | Done | This document |
| Architecture and diagrams | Done | `docs/ARCHITECTURE.md`, `docs/images/` |
| Data dictionary | Done, telecom and IEEE-CIS | `docs/04-data-dictionary.md` |
| End-to-end guide | Done | `docs/05-end-to-end-guide.md` |
| Progress log | Done | `docs/PROGRESS.md` |
| Model card | Done | Appendix A |
| Operations runbook | Done | Appendix B |
| README with quickstart | Done | `README.md` |
| Security note | Done | `SECURITY.md`, and `.env.example` |
| LICENSE | Done | `LICENSE` (MIT) |
| Pending items | Done | `docs/PENDING.md` |
| API reference | Partial | FastAPI serves interactive docs at `/docs`; no written guide |
| Dashboard screenshot | Replace | `docs/images/dashboard-example.png` is an older example; replace it with one from your own run |
| Grafana screenshots | Missing | Save the API and agents dashboards as `docs/images/grafana-dashboard.png` and `docs/images/grafana-agents.png` |
| Change log | Not needed | The merge commits on `main` and `docs/PROGRESS.md` record each phase |
