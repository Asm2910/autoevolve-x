"""Prometheus metrics for the agents and experiments.

These live in their own registry, separate from the API's. Missions are batch jobs that start
and finish, so a scraper would usually miss them; they are pushed to a Pushgateway (added in
the observability phase, see observability/push.py). The same numbers are also available through
`snapshot()` and in the JSONL trace log.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

AGENT_REGISTRY = CollectorRegistry()

AGENT_EXECUTIONS = Counter(
    "agent_execution_total", "Agent steps executed", ["agent", "status"], registry=AGENT_REGISTRY
)
AGENT_DURATION = Histogram(
    "agent_execution_duration_seconds",
    "Wall-clock time of an agent step",
    ["agent"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 900),
    registry=AGENT_REGISTRY,
)
AGENT_FAILURES = Counter(
    "agent_failures_total", "Agent steps that failed", ["agent", "error"], registry=AGENT_REGISTRY
)
LLM_REQUESTS = Counter("llm_requests_total", "Language-model calls", ["agent", "model"], registry=AGENT_REGISTRY)
LLM_TOKENS = Counter(
    "llm_tokens_total", "Language-model tokens", ["agent", "direction"], registry=AGENT_REGISTRY
)
LLM_COST = Counter(
    "llm_cost_usd_total", "Estimated language-model spend in US dollars", ["agent"], registry=AGENT_REGISTRY
)
EXPERIMENTS = Counter(
    "experiments_total", "Experiments run by the agent, by verdict", ["verdict"], registry=AGENT_REGISTRY
)
EVAL_FINDINGS = Counter(
    "evaluation_findings_total", "Findings reported by the evaluation swarm", ["agent", "severity"],
    registry=AGENT_REGISTRY,
)
BEST_PR_AUC = Gauge(
    "agent_best_validation_pr_auc", "Validation PR-AUC of the current champion", registry=AGENT_REGISTRY
)
AGENT_RUN_LAST = Gauge(
    "agent_run_last_timestamp_seconds", "Unix time the last agent script finished", registry=AGENT_REGISTRY
)
AGENT_RUN_SECONDS = Gauge(
    "agent_run_duration_seconds", "Wall-clock time of the last agent script", registry=AGENT_REGISTRY
)
MISSION_ROUNDS = Gauge(
    "mission_rounds_completed", "Improvement rounds finished in the last mission", registry=AGENT_REGISTRY
)
MISSION_HALTED = Gauge(
    "mission_halted", "1 if the last mission was stopped by the budget or kill switch", registry=AGENT_REGISTRY
)


def snapshot() -> dict[str, float]:
    """Current value of every sample, keyed by 'name{labels}'. Handy for tests and reports."""
    values: dict[str, float] = {}
    for family in AGENT_REGISTRY.collect():
        for sample in family.samples:
            labels = ",".join(f"{k}={v}" for k, v in sorted(sample.labels.items()))
            values[f"{sample.name}{{{labels}}}" if labels else sample.name] = sample.value
    return values
