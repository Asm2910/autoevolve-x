"""One structured record for every agent step, on disk and in the metrics.

`Observer.step(...)` is a context manager. It times the step, writes a JSON line to the trace
file (when one is configured) and updates the Prometheus counters. The trace is how a finished
run is audited or replayed: it says which step ran, how long it took, what it decided and why
it failed.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from observability import agent_metrics as metrics


class TraceLog:
    def __init__(self, path: Path | None = None, run_id: str | None = None) -> None:
        self.path = Path(path) if path else None
        self.run_id = run_id or uuid.uuid4().hex[:12]
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, **fields) -> dict:
        entry = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), "run_id": self.run_id, **fields}
        if self.path:
            with open(self.path, "a", encoding="utf-8") as file:
                file.write(json.dumps(entry, default=str) + "\n")
        return entry

    @staticmethod
    def load(path: Path) -> list[dict]:
        with open(path, encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]


class Observer:
    """Times steps and reports them to the trace log and to Prometheus."""

    def __init__(self, trace: TraceLog | None = None) -> None:
        self.trace = trace or TraceLog(None)

    @contextmanager
    def step(self, agent: str, **meta) -> Iterator[dict]:
        """Wrap one step. The yielded dict can be filled with outcome details (verdict, tokens)."""
        outcome: dict = {}
        started = time.perf_counter()
        status, error = "ok", ""
        try:
            yield outcome
        except Exception as exc:
            status, error = "error", type(exc).__name__
            metrics.AGENT_FAILURES.labels(agent, error).inc()
            raise
        finally:
            duration = time.perf_counter() - started
            metrics.AGENT_EXECUTIONS.labels(agent, status).inc()
            metrics.AGENT_DURATION.labels(agent).observe(duration)
            self.trace.record(
                agent=agent, status=status, error=error, duration_seconds=round(duration, 4), **meta, **outcome
            )
