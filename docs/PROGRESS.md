# Build progress (v3 upgrade)

The goal of v3 is to implement the originally planned architecture: a model arena, parallel
execution, specialised agents with guardrails, a utility-score judge, a real-data check and
agent-level observability. Each phase is merged into `main` on its own.

| Phase | Branch | Status | What it adds |
| --- | --- | --- | --- |
| 10 | `phase10-model-arena` | done | Eight algorithms behind one registry, a fair validation-only arena with paired confidence intervals, fresh data (v2) |
| 11 | `phase11-ray-engine-agent-platform` | planned | Ray executor, guardrails, harness, context engineering |
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

## Operational notes

- Training the seven serving candidates takes about 6 minutes on a two-core machine (random forest
  and extra trees dominate). CI trains three fast models through `AUTOEVOLVE_CANDIDATES`.
- LightGBM needs the OpenMP runtime. On macOS: `brew install libomp`.
- The Docker image installs CatBoost (about 270 MB) only when CatBoost is the served model.
