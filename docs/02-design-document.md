# AutoEvolve-X: Design Document

2026-10-18

AutoEvolve-X lets a team of small agents help improve a fraud-risk model without ever being able to cheat on the evaluation. A language model may propose ideas. Plain code runs them, measures them and decides. This document explains the design choices and why they were made. It covers the whole system through version 2: the mission planner, the data and evaluation swarms, the model arena and judge, the guardrails, the research agent and the monitoring.

## Goals and non-goals

Goals:

- Improve a rare-event classifier through agent-proposed experiments, judged only on validation data.
- Review the data and the model with specialised agents before and after any change, and report what they find.
- Choose the served model by cost and evidence, not by the highest single number.
- Bound every agent: a budget, a kill switch, schema-checked inputs and outputs, and a trace of what it did.
- Make every decision reproducible and auditable from a single memory database.
- Serve the champion through an API that is monitored for traffic, speed, errors and score drift, and show what the agents did on a second dashboard.
- Run the whole system from one repository, with CI that proves it still works.

Non-goals:

- No autonomous deployment: no agent promotes a model to the API by itself.
- No real customer data, and no claim of production-grade accuracy.
- No large-scale AutoML search; a mission runs a handful of experiments.
- No data ingestion: the pipelines have no live feed.
- No Kubernetes, Kafka, Spark, Airflow, GPUs, language-model fine-tuning, or more than a dozen agents.
- No production authentication.

## Design principles

The central rule is that the language model never decides anything that matters. It suggests, and deterministic code checks.

1. **Propose versus decide.** Models and agents propose. Python runs the work, measures it and applies fixed rules.
2. **A sealed test set.** Agents can load only train and validation data. The test split is read once per dataset, and a guard in memory refuses a second sealed-test run.
3. **Uncertainty is part of the answer.** Every score carries a bootstrap confidence interval. Adoption needs the lower bound above zero. Model choice treats models whose gap interval includes zero as tied.
4. **Many tests need a correction.** Many features, many segments and many experiments mean some results look real by chance. Feature screening uses Benjamini-Hochberg false discovery control and an effect-size floor. Segment findings use a Bonferroni correction.
5. **Memory has one source of truth.** SQLite holds every experiment and lesson. The vector index is derived from it and can be rebuilt.
6. **Fail soft, stop hard.** A failed step is recorded and the mission carries on. A budget overrun or the kill switch stops the mission cleanly, and it still reflects before it ends.
7. **Outside text is data, not instructions.** Anything that did not come from the code is sanitised and delimited before it reaches a prompt.
8. **Rehearsals leave no trace.** Simulated drift checks are never written to memory, so they cannot teach the agents false lessons.
9. **Say what is not claimed.** Reports and documents state their own limits.

## System architecture

![System architecture](images/system-architecture.png)

Data flows from two sources, kept apart, into one feature and split step. The mission planner sends work to the swarms, the improvement agent and the arena. Reflection writes grounded lessons to one memory, which the research agent and the control panel read. The judge's served model goes to the API, which Prometheus watches. A harness wraps every agent, and a sealed test set sits behind all of it. [ARCHITECTURE.md](ARCHITECTURE.md) has the same picture as editable diagrams.

## Data and evaluation design

Two datasets, never combined.

| Dataset | What it is | Split | Positive rate |
| --- | --- | --- | --- |
| Telecom v2 | Synthetic world with hidden fraud, so the answer is known and the pipeline can be tested honestly. 254 features in five groups | Nine monthly snapshots: 5 train, 2 validation, 2 test | About 0.8% of test rows |
| IEEE-CIS | Real e-commerce card fraud (Kaggle). A 300,000-row random sample keeping time order. 440 features in 11 groups, built by `data_adapters/ieee_cis.py` | By time: earliest 60% train, next 20% validation, last 20% sealed test | 3.4% of test rows |

IEEE-CIS is used as validation on real fraud data. It is e-commerce fraud, not telecom, and it is not multi-source telecom data. Only the labelled train files of the Kaggle competition are used, with our own temporal split. A causal `history` group is built per card from earlier payments only, so computing it before the split does not leak.

A temporal split matters because behaviour changes over time, and a random split would let the model see the future. With rare positives the headline metric is PR-AUC rather than accuracy or ROC-AUC. Each score gets a bootstrap confidence interval, and comparisons use a paired bootstrap on the same rows.

A switch (`AUTOEVOLVE_DATASET=ieee`) moves every path, config, model folder and memory file to the IEEE copy, so one dataset can never overwrite the other.

## Mission design

A mission is a LangGraph graph with three kinds of node: plan, dispatch and reflect.

![Mission graph](images/mission-graph.png)

- **Planner.** Rule-based. The depth picks the steps: quick (data review, evaluate, improve, reflect), standard and thorough (adding the model arena and more experiments and rounds). The planner does not read the goal text. The goal is recorded in the report. This is stated as a limit, not hidden.
- **Re-planning.** After the data review and the evaluation the planner revises the remaining steps. Ideas from the swarms are passed to the improvement step.
- **Dispatch.** Runs the next pending step. A failed step is recorded and removed from the queue, so a broken step cannot loop forever.
- **Reflection.** Writes lessons that are grounded in what the mission actually found, and decides whether to run another round. It does so only if the last round adopted a change and rounds remain.
- **Stopping.** The budget and the kill switch stop the mission, which still reflects.

## Swarm design

A swarm is a small set of deterministic agents on the shared harness. They do not call a language model. They exist so that specific questions are asked every time, by code that can be tested.

| Swarm | Agents | What they ask |
| --- | --- | --- |
| Data | Schema, Quality, Relationship, Feature | Right columns and types? Missing values, duplicates, constants, impossible values? Do sources join cleanly, and is history before each snapshot? Which feature groups carry signal after a multiple-testing correction, and is anything leaking? |
| Evaluation | Performance, Segment, Error, Drift | How good is the model, with intervals, and is it steady over time? Where is it weaker than overall, after Bonferroni correction? How do missed frauds differ from caught ones? Has the data being scored stopped looking like the training data? |

The segment agent shows only the top shortfalls plus a summary line, and merges groups that cover identical accounts, so a report does not list 100 near-duplicates. The feature agent flags a feature that predicts almost perfectly (AUC of 0.97 or more) as a possible leak.

## Model arena and judge design

Eight algorithms sit behind one registry: logistic regression, random forest, extra trees, XGBoost, LightGBM, scikit-learn histogram gradient boosting, CatBoost and isolation forest. They are all trained on the same data and ranked on validation.

The arena runs sequentially or on Ray. On a 10-core laptop Ray gave no speed-up for seven small models (0.93x with two workers, 0.79x with four), because starting processes and moving data cost more than training. Ray is kept for isolation, per-model timeouts, retries and a path to a cluster, and the result is documented as it is.

The judge picks the served model. Rules, in order:

1. Models that cannot explain a score are excluded (random forest, extra trees, histogram gradient boosting, logistic regression). The explainable set is XGBoost, LightGBM and CatBoost.
2. Each model gets a business utility per 10,000 accounts: 400 for each fraud caught, minus 6 for each reviewed account, minus 3 for each false alarm. The prices are assumptions in the config.
3. A paired bootstrap compares each model with the best. A model whose gap interval includes zero is tied.
4. Among tied models the judge serves the cheapest to retrain, then the smallest, unless it is more than 10% below the best utility.

On telecom three boosted models were tied and XGBoost was served because it trains fastest. `training.selection: pr_auc` restores the older pick-the-highest rule.

## Improvement agent design

The improvement agent is a LangGraph loop with six nodes: recall, propose, novelty, run, decide, record.

![Agent loop](images/agent-loop.png)

- **Proposers.** Claude proposes through structured output validated against a schema. A fallback switches to a rule-based proposer if the model call fails. Evaluation and data findings reach the proposer as delimited, sanitised data.
- **Novelty check.** A proposal that matches a past experiment by configuration is skipped.
- **Decision rule.** Adopt only if the paired bootstrap lower bound is above zero and the mean gain is at least 0.01 PR-AUC.
- **Stopping.** The model may stop early. The budget and kill switch can stop it at any step.

The rule is strict on purpose. On the telecom data the agent tried window changes and rejected them all, because each interval included zero or sat below it. On IEEE-CIS it adopted deeper trees, with a validation gain of +0.018 at depth 4 and an interval above zero.

## Harness and guardrail design

Every agent, including the research agent, runs inside the same harness.

![Agent harness](images/agent-harness.png)

| Part | What it does |
| --- | --- |
| Contract | Name, responsibility, input and output schema, allowed tools, timeout and retries |
| Budget | Caps on language-model calls, input and output tokens, cost, wall time and experiments. Charged as work happens |
| Kill switch | A file (`memory/STOP`). When it exists the next check stops the run |
| Context builder | Ranks evidence and fits it into a token budget, so a prompt stays small |
| Untrusted-text guard | Sanitises and delimits any text that came from outside the code, and replaces text that looks like an instruction to the model |
| Observer | A JSONL trace for each run and Prometheus metrics for steps, failures, tokens, cost, experiments and findings |

The injection filter is pattern-based. It is one layer of defence, not a guarantee. The stronger protection is structural: model output is validated against a schema, and nothing a model returns can change a model, a threshold or a score without passing a check written in code.

## Memory design

Memory has two stores with different jobs. SQLite is the source of truth, and ChromaDB is a search index built from it.

- **Tables.** One for experiments and one for lessons, plus an FTS5 table for keyword search.
- **Hybrid search.** Keyword and vector results are merged by reciprocal rank fusion. Embeddings come from a hashing embedder or MiniLM.
- **Idempotent writes.** An experiment is identified by kind, name, configuration hash and dataset hash. Recording it twice changes nothing.
- **Per dataset.** The IEEE dataset has its own database and index.
- **Kinds of record.** Agent experiments, the final test, drift checks, and rows ingested from training and ablation runs.

The same table backs the final-test guard: if a confirmation already exists for a dataset hash, the confirmation script refuses to run again.

## Research agent design

The research agent answers questions about the project from the project's own README, docs, reports, lessons and experiments. It retrieves passages by BM25 word ranking, optionally fused with embeddings. It has two modes.

- **Without a model:** it returns the closest passages, each with its source.
- **With a model:** the model must answer only from numbered passages and cite them. Code then checks the answer. A citation must name a passage the model was shown. Every number must appear in a cited passage, or it is listed as ungrounded. An answer with no citation is withheld. If nothing is retrieved, it says it could not find the answer.

Passages and the question are sanitised as untrusted text, so a note in a report that says "ignore your instructions" is replaced with a placeholder.

## Serving and observability design

The champion is served by a small FastAPI service in a non-root Docker image. Prometheus scrapes it every 5 seconds and Grafana shows it.

- **Endpoints.** `/predict` scores one subscriber and returns the top reasons. `/health`, `/model/info` and `/metrics` cover status, model details and Prometheus.
- **Metrics and labels.** Request count and latency, prediction counts, rejected requests, the served model and threshold, and a histogram of predicted scores. Labels use the route pattern, so a scanner cannot create endless series.
- **Live drift signal.** The score histogram shows when scores move up, with no labels.
- **Batch jobs push.** Missions and agent runs finish before Prometheus would scrape them. They push their final metrics to a Pushgateway, which keeps the last run of each script for each dataset. Pushing is off unless a gateway is configured, and a failure never fails a run.
- **Two dashboards.** One for the API, one for the agents: time since the last run, champion score, language-model spend and tokens, experiments by verdict, failed steps, findings by severity, step latency and calls by agent.
- **Alerts.** Eight rules: five for the API (down, high error rate, slow predictions, score shift, rejected requests) and three for the agents (failed steps, spend above 1.50 US dollars, no report for a week).
- **MLflow.** Training, arena and agent runs are logged. A compose MLflow server is available.

![Monitoring flow](images/monitoring-flow.png)

## Control panel design

A Streamlit app with three parts: the review page (ledger, final test, importance, drift), mission control and an ask page. Mission control starts a mission as a separate process, so closing the tab does not stop it. It watches the log, and stops the run by creating the kill-switch file. Only one mission runs at a time. The page cannot start a mission with invalid input, because it asks the planner to validate the plan first.

## Key decisions and trade-offs

| Decision | Chosen | Alternative | Why |
| --- | --- | --- | --- |
| Who adopts a change | Fixed rule on the confidence interval | Let the model judge | A rule is auditable and cannot be talked into noise |
| Who picks the served model | Judge: utility, ties, cost | Highest PR-AUC | Differences between top models are inside noise, so cost to run is a fair tie-breaker |
| Swarm agents | Deterministic code | Language-model agents | Every question is asked every time, and it is testable and free |
| Planner | Rule-based from the depth | Model-written plan | A plan the code validates cannot invent steps. The cost is that it ignores goal text |
| Feature screening | False discovery control plus an effect floor | Raw p-values | With 100,000 or more rows almost everything is significant |
| Parallel training | Ray, with sequential as default | Ray only | Ray gave no speed-up here, and sequential is simpler to debug |
| Real-data check | IEEE-CIS, kept separate | Mix with telecom | Mixing would make a claim the data cannot support |
| Retrieval | BM25, optional embeddings | Embeddings only | Exact terms (model names, metric names) matter in this corpus |
| Batch metrics | Pushgateway | Scrape the script | A mission ends before it can be scraped |
| Headline metric | PR-AUC | Accuracy or ROC-AUC | Positives are rare |
| Split | By time | Random | A random split leaks the future |
| Drift detection | PSI and KS on features | Track live accuracy | Fraud labels arrive late |
| Alert delivery | Rules only, no Alertmanager | Add notifications | Kept in scope; listed as a gap |

## Risks and limitations

| Risk | Mitigation | Gap that remains |
| --- | --- | --- |
| Agents overfit validation | Strict adoption rule, sealed test | Experiments share one validation set with no correction for multiple comparisons |
| Synthetic data flatters the model | Known ground truth, honest caveats, IEEE-CIS check | Real data is one e-commerce dataset, sampled to 300,000 rows |
| The planner ignores the goal | Stated in the report, docs and UI | No natural-language planning |
| Prompt injection through data or reports | Sanitise, delimit, schema-check outputs | The filter is pattern-based |
| The language model returns bad output | Schema validation, rule-based fallback, citation and number checks | Live-model paths were tested with scripted doubles, so a real-key run is still to be read by a person |
| Scores are not true probabilities | Threshold chosen on validation | No calibration step |
| Ray does not help at this size | Documented, sequential default | Benefit unproven for larger models |
| Drift is detected but not acted on | Alert and a recommendation | No ingestion or automatic retrain |
| Alerts have nowhere to go | Visible in Prometheus and Grafana | No Alertmanager |
| The agents dashboard shows only the last run | Traces and memory hold history | No time series of runs |
| Docker only partly tested | CI builds the image | The compose stack and the CatBoost-served path were checked by hand only |

## Roadmap and open questions

- Let the agent propose new features, not only model settings. That would widen what it can find and the risk of leakage.
- Correct for multiple comparisons across experiments that share one validation set.
- Add calibration, Alertmanager and a data-ingestion step.
- Run the arena on a larger dataset to see whether Ray starts to help.
- Give the planner a way to read the goal, behind the same validation as today.
