"""Where the arena's training jobs run: one after another, or in parallel on Ray.

Both executors take the same input and return results in the same order, so the rest of
the system does not care which one is used.

Ray notes (the parts worth knowing):
  * The training and validation tables are put in Ray's shared object store once. Every
    task receives a reference, not a copy of 100k rows.
  * Each task asks Ray for one CPU, so at most `workers` models train at the same time.
  * Each model is also told to use only its share of the machine's cores. Without this,
    four models that each grab every core would fight each other and run slower.
  * A task that crashes its worker process is retried once. A task that raises is caught
    inside `train_candidate` and becomes a failed result. A task that exceeds the time
    limit is cancelled and reported as failed. One bad model never stops the others.
  * If Ray is not installed or cannot start, the arena falls back to sequential.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import replace

from arena.runner import ArenaData, ArenaResult, ArenaTask, describe, train_candidate


class SequentialExecutor:
    name = "sequential"
    workers = 1

    def run(
        self,
        tasks: list[ArenaTask],
        data: ArenaData,
        review_fraction: float,
        log: Callable[[str], None] = print,
    ) -> list[ArenaResult]:
        results = []
        for task in tasks:
            result = train_candidate(task, data.train_x, data.train_y, data.val_x, data.val_y, review_fraction)
            log(describe(result))
            results.append(result)
        return results

    def close(self) -> None:
        return None


class RayExecutor:
    name = "ray"

    def __init__(self, workers: int, timeout_seconds: float | None = None) -> None:
        import ray

        self.ray = ray
        self.workers = max(1, int(workers))
        self.timeout_seconds = timeout_seconds
        self.threads_per_task = max(1, (os.cpu_count() or 1) // self.workers)
        self._started_here = not ray.is_initialized()
        if self._started_here:
            ray.init(
                num_cpus=self.workers,
                include_dashboard=False,
                logging_level=logging.ERROR,
                ignore_reinit_error=True,
            )

    def run(
        self,
        tasks: list[ArenaTask],
        data: ArenaData,
        review_fraction: float,
        log: Callable[[str], None] = print,
    ) -> list[ArenaResult]:
        ray = self.ray
        shared = [ray.put(x) for x in (data.train_x, data.train_y, data.val_x, data.val_y)]
        remote = ray.remote(num_cpus=1, max_retries=1)(train_candidate)
        refs = {
            remote.remote(replace(task, threads=self.threads_per_task), *shared, review_fraction): i
            for i, task in enumerate(tasks)
        }
        results: dict[int, ArenaResult] = {}
        deadline = None if self.timeout_seconds is None else time.monotonic() + self.timeout_seconds
        pending = list(refs)
        while pending:
            wait = None if deadline is None else max(0.0, deadline - time.monotonic())
            ready, pending = ray.wait(pending, num_returns=1, timeout=wait)
            if not ready:  # time limit reached with work still running
                for ref in pending:
                    ray.cancel(ref, force=True)
                    index = refs[ref]
                    results[index] = ArenaResult(
                        tasks[index].model, False, error=f"TimeoutError: exceeded {self.timeout_seconds:.0f}s"
                    )
                    log(describe(results[index]))
                break
            for ref in ready:
                index = refs[ref]
                try:
                    results[index] = ray.get(ref)
                except Exception as exc:  # noqa: BLE001 - e.g. the worker process was killed
                    results[index] = ArenaResult(
                        tasks[index].model, False, error=f"{type(exc).__name__}: {str(exc)[:160]}"
                    )
                log(describe(results[index]))
        return [results[i] for i in range(len(tasks))]

    def close(self) -> None:
        if self._started_here and self.ray.is_initialized():
            self.ray.shutdown()


def make_executor(workers: int, timeout_seconds: float | None = None, log: Callable[[str], None] = print):
    """Ray when more than one worker is requested and Ray works; otherwise sequential."""
    if workers <= 1:
        return SequentialExecutor()
    try:
        return RayExecutor(workers, timeout_seconds)
    except ImportError:
        log("Ray is not installed (pip install ray); running sequentially.")
    except Exception as exc:  # noqa: BLE001 - any start-up failure falls back to sequential
        log(f"Ray could not start ({type(exc).__name__}: {str(exc)[:120]}); running sequentially.")
    return SequentialExecutor()
