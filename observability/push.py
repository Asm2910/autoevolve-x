"""Send the agent metrics to a Pushgateway when a script finishes.

Missions and agent runs are short batch jobs: Prometheus would usually scrape between them and see
nothing. So each script pushes its final numbers to a Pushgateway, which keeps them for Prometheus.

  - Off unless a gateway is named: set PUSHGATEWAY_URL (for example http://localhost:9091) or
    `agent_platform.pushgateway_url` in the config. Nothing is sent otherwise.
  - Never raises: a missing or slow gateway must not fail a mission. It prints one line and carries on.
  - One group per (dataset, script), so a mission does not overwrite the data review, and telecom does
    not overwrite ieee. A push replaces that group, so the dashboard shows the LAST run of each script,
    not a history. History lives in the JSONL traces and the experiment memory.
"""

from __future__ import annotations

import atexit
import os
import time

from prometheus_client.exposition import push_to_gateway

import datasets
from observability import agent_metrics as metrics

JOB = "autoevolve-agents"


def gateway_url(config: dict | None = None) -> str:
    return (os.environ.get("PUSHGATEWAY_URL") or (config or {}).get("pushgateway_url") or "").strip()


def push_metrics(script: str, started: float | None = None, url: str | None = None, timeout: float = 5.0) -> bool:
    """Push the agent registry. Returns True if it was sent; False if disabled or the gateway failed."""
    target = url if url is not None else gateway_url(_platform_config())
    if not target:
        return False
    now = time.time()
    metrics.AGENT_RUN_LAST.set(now)
    if started is not None:
        metrics.AGENT_RUN_SECONDS.set(now - started)
    try:
        push_to_gateway(target, job=JOB, registry=metrics.AGENT_REGISTRY,
                        grouping_key={"dataset": datasets.active(), "script": script}, timeout=timeout)
    except Exception as err:  # noqa: BLE001 - a monitoring problem must never fail the run
        print(f"(metrics not pushed to {target}: {type(err).__name__}: {err})")
        return False
    print(f"(metrics pushed to {target} as {JOB}/{datasets.active()}/{script})")
    return True


def push_at_exit(script: str) -> None:
    """Register a push for when the script ends, however it ends."""
    started = time.time()
    atexit.register(push_metrics, script, started)


def _platform_config() -> dict:
    try:
        from models.training import load_config

        return load_config().get("agent_platform", {})
    except Exception:  # noqa: BLE001
        return {}
