# Build progress (v3 upgrade)

The goal of v3 is to implement the originally planned architecture: a model arena, parallel
execution, specialised agents with guardrails, a utility-score judge, a real-data check and
agent-level observability. Each phase is merged into `main` on its own.

| Phase | Branch | Status | What it adds |
| --- | --- | --- | --- |
| 10 | `phase10-model-arena` | done | Eight algorithms behind one registry, a fair validation-only arena with paired confidence intervals, fresh data (v2) |
| 11 | `phase11-ray-agent-platform` | done | Ray executor with benchmark; budgets, kill switch, guardrails, harness, context engineering, traces and metrics |
| 12 | `phase12-eval-swarm` | planned | Performance, segment, error and drift agents |
| 13 | `phase13-judge-utility` | planned | Utility-score judge |
| 14 | `phase14-data-swarm` | planned | Schema, quality, relationship and feature agents |
| 15 | `phase15-planner-reflection` | planned | Planner, ML manager and reflection in one graph |
| later | | planned | Research agent, mission UI, observability v2, IEEE-CIS real dataset, final documents |

## Phase 10: model arena

**Dataset v2.** `configs/data.yaml` now uses seed 20261018. The old sealed test set (v1, seed 42)
was already scored once, so a new architecture that might pick a different champion needs a new
one. The v1 results stay valid as history: champion PR-AUC 0.272 on the v1 test set.

**Registry.** `arena/registry.py` is the single list of algorithms: logistic regression, random
forest, extra trees, XGBoost, LightGBM, HistGradientBoosting, CatBoost and an unsupervised
isolation forest. It also holds each model's hyperparameter bounds and how its inputs must be
prepared. The experiment spec and the agent prompt read their allowed models and bounds from it.

**Arena.** `python -m scripts.run_arena` trains every model on the same rows and scores them on the
validation snapshots only. Bootstrap resamples are shared across models, so the ranking is paired.
A model is reported as "tied with best" when its interval overlaps the leader.

**First result on dataset v2 (validation, 484 positives).** Six of eight models are statistically
tied with the leader (CatBoost 0.303, XGBoost 0.302, random forest 0.294, HistGradientBoosting 0.292,
LightGBM 0.291, extra trees 0.287). Logistic regression (0.256) and the unsupervised isolation
forest (0.128) are clearly worse. The signal is real but the data cannot separate the strong models,
which is why a cost-aware judge is added in phase 13.

**Known limitation.** `train_models` still serves the model with the best validation PR-AUC,
even when several are tied. The utility-score judge in phase 13 replaces that rule.

## Phase 11: Ray and the agent platform

**Ray.** `python -m scripts.run_arena --workers 4` trains several models at once. The data goes into
Ray's shared object store once; each task takes one CPU and is told to use only its share of the
cores, so parallel models do not fight each other. A crashed worker is retried once, a model that
raises becomes a failed result, a task over the time limit is cancelled, and if Ray is missing the
arena runs sequentially. `python -m scripts.benchmark_ray` times 1, 2 and 4 workers and checks that
the scores agree. On my 2-core sandbox the speed-up was only 1.07x, because the machine had no
spare cores; the benchmark on your Mac is the meaningful one.

**Agent platform (`agentkit/`).**
- `budget.py`: limits on calls, tokens, cost, time and experiments, plus a kill switch file.
- `guardrails.py`: sanitising and delimiting untrusted text, with an instruction-injection filter.
- `data_access.py`: the only door agents use to read data; the test split is refused.
- `harness.py`: a base agent with a typed contract, timeout, retries, a tool allow-list and traces.
- `tracing.py` and `observability/agent_metrics.py`: a JSONL trace and Prometheus metrics per step.
- `context.py`: token-budgeted prompts, history compaction, per-agent context policy, prompt versions.
- `llm.py`: token accounting, and recording, replay and scripted model doubles.

**Evaluations.** `tests/agent_evals/` runs whole missions with a scripted model: limits are
respected, repeated and out-of-bounds proposals are refused, hostile text in memory is neutralised,
the test split is unreachable, and a recorded mission replays to identical decisions.

**Limits to know.** The text filter is pattern-based and can be evaded; the structural controls
are what make an injection harmless. A step that times out stops being waited for but its thread
cannot be killed, so long work should run in Ray tasks. Metrics are collected in their own registry
and will be pushed to Prometheus in the observability phase.

## Phase 12: evaluation swarm

Four deterministic agents in `swarm/evaluation.py`, each built on the agent harness (typed contract,
time limit, trace record) and each answering one question about a model's validation scores:

- **Performance:** PR-AUC with a bootstrap interval, precision and recall at the review cut-off, lift,
  and whether quality holds across time periods.
- **Segment:** recall at the review cut-off for every customer group large enough to judge. A group is
  flagged only when the upper end of its interval stays below the overall recall, with the threshold
  tightened for the number of groups tested (Bonferroni), so a long list of groups does not produce
  false alarms.
- **Error:** which features separate the frauds the model misses from the ones it catches, as
  standardised differences, summarised by feature group.
- **Drift:** population stability between training and validation data, reusing `monitoring/drift.py`.

`EvaluationSwarm` runs all four, contains a failure in any one, ranks the findings by severity and
writes `evaluation_report.{md,json}`. Warnings become short hypotheses saved to
`memory/evaluation_findings.json`; `run_agent --use-findings` passes them to the proposer as
untrusted evidence (sanitised and delimited like any outside text). Finding counts are exported as
`evaluation_findings_total`. The sealed test split is not reachable from this package.

Real validation result for the baseline: no underserved group, no drift, steady quality over time,
and missed fraud differs most in the usage group (fewer distinct destinations and weaker
international-call trends than the frauds that were caught).

## Phase 13: the judge and utility score

`judge/utility.py` replaces "highest validation PR-AUC wins" with a written policy, run by
`scripts/train_models.py` (`training.selection: utility`; `pr_auc` restores the old rule):

1. Hard constraint: a model that cannot explain its scores is excluded, because the API returns reasons.
2. Utility per 10,000 accounts at the review cut-off: fraud caught x value, minus reviewed x review cost,
   minus false alarms x friction cost. The prices are assumptions in `configs/models.yaml` (`judge:`).
3. A paired bootstrap on identical resampled accounts gives each model's gap to the best; models whose
   interval includes zero are tied.
4. Among ties, the cheapest to operate is served (fastest to retrain, then smallest), unless its point
   utility is more than 10% below the best.

Real result on validation: CatBoost has the highest utility (15,968), XGBoost (15,881) and LightGBM
(15,708) are tied with it, so XGBoost is served: it trains fastest and does not need the 270 MB CatBoost
install in Docker. Four models were excluded for lacking explanations. The verdict, intervals and reasons
are written to `data/processed/judge_report.md` and stored in `model_metadata.json`.
Because fixed review capacity makes utility track recall at the cut-off, the judge's main value is the
explicit prices, the tie handling and the operating-cost rule, not a different ranking of quality.

## Phase 14: data swarm

`swarm/data.py` adds four more agents, built like the evaluation agents and generic over any set of tables
(`DataContext` plus a `TableSpec` per table), so the same agents will check the real-data adapter:

- **Schema:** expected columns, timestamp and numeric types, allowed category values.
- **Quality:** missing values (sampled), duplicate ids, constant columns, negative values where impossible,
  timestamps outside the data window.
- **Relationships:** rows with no matching customer, share of customers present in each source, hot keys,
  enough history before the first snapshot, duplicate or empty labels.
- **Features:** tests every training feature against the label and corrects for the number of tests
  (Benjamini-Hochberg, 5% false discovery rate). A feature counts only if it survives the correction and
  its effect is large enough to matter (|AUC - 0.5| of at least 0.03, or Cramer's V for categories); with
  100k+ rows even trivial effects are "significant". It ranks feature groups and windows, proposes
  groups to enrich or test dropping, and flags any single feature that separates the classes almost
  perfectly as possible leakage.

`python -m scripts.run_data_review` writes `data_review_report.{md,json}` and `memory/data_findings.json`;
`run_agent --use-findings` reads both swarms' ideas as untrusted evidence. The feature agent reads the
training split only.

Real result on the telecom data: schemas and quality are clean. One real finding: only 55% of customers
have any risk event, so the risk group partly encodes "has a risk record". Usage is the strongest
group (57 of 63 features carry signal), then risk, recharge, network, and customer last (1 of 8).

## Phase 15: real data (IEEE-CIS)

The same pipeline now runs on real e-commerce card fraud: IEEE-CIS (Vesta, via Kaggle), 590,540 payments,
3.5% fraud, 183 days, two linked sources (payments and device/identity, joined on TransactionID; identity exists
for 24% of payments). Only the two labelled *train* files are used, and the project makes its own temporal split
(earliest 60% of time trains, next 20% tunes, last 20% is the sealed test). Public multi-source *telecom* fraud
data does not exist, so this is validation on real fraud data, not a replacement for the telecom work.

- `data_adapters/ieee_cis.py`: reads the Kaggle zip or folder, builds 440 features in 11 groups using the
  project's `<group>__<metric>` naming (transaction, card, address, email, counts, timedeltas, match, vesta,
  identity, device) plus a `history` group made from each card's earlier payments only (count so far, amount
  against the card's own average, seconds since its previous payment), so nothing from the future leaks.
  Rare categories are collapsed using training rows only.
- `datasets.py`: `AUTOEVOLVE_DATASET=ieee` switches every script to its own processed folder, configuration
  (`configs/ieee.yaml`), model folder and agent memory. The telecom data and the served model are never touched.
- `scripts/prepare_ieee.py`: one command from the Kaggle zip to the processed files (default 300,000 rows
  spread evenly over time; `--sample 0` keeps all).
- Data swarm: `ieee_specs()` plus a summary for sparse columns, so 282 mostly-empty columns become one finding.
- Evaluation swarm: shows only the largest segment shortfalls and summarises the rest.

Real results (300,000-row sample, validation / sealed test):
- Validation PR-AUC: LightGBM 0.558, XGBoost 0.550, logistic regression 0.466 (ROC-AUC about 0.90).
- Sealed test PR-AUC (a later period): LightGBM 0.448, XGBoost 0.431. Performance falls in the later period, which the
  telecom data also showed; fraud patterns change over time.
- The judge serves XGBoost (tied with LightGBM within noise, cheaper to train).
- Evaluation swarm: the model catches only 6% of fraud in product W (1,044 cases) against 38% overall, so
  most missed fraud sits in the largest product line.
- The self-improving agent adopted a change on real data: deeper trees, validation PR-AUC +0.023
  (95% interval +0.019 to +0.026).

## Operational notes

- Training the seven serving candidates takes about 6 minutes on a two-core machine (random forest
  and extra trees dominate). CI trains three fast models through `AUTOEVOLVE_CANDIDATES`.
- LightGBM needs the OpenMP runtime. On macOS: `brew install libomp`.
- The Docker image installs CatBoost (about 270 MB) only when CatBoost is the served model.
