"""Sequential and Ray executors must agree, and failures must stay contained."""

import os

import numpy as np
import pandas as pd
import pytest

from arena import executor as executor_module
from arena.executor import SequentialExecutor, make_executor
from arena.runner import ArenaData, ArenaTask, run_arena

NUMERIC = ["a", "b", "c"]
CATEGORICAL = ["plan"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    n = 1200
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
            "plan": rng.choice(["x", "y", "z"], n),
        }
    )
    y = (rng.random(n) < 1 / (1 + np.exp(-(2.0 * frame["a"].to_numpy() - 2.5)))).astype(int)
    return ArenaData(frame.iloc[:800], y[:800], frame.iloc[800:], y[800:])


TASKS = [
    ArenaTask("logistic_regression", {"C": 0.1, "max_iter": 200}, NUMERIC, CATEGORICAL, 1),
    ArenaTask("random_forest", {"n_estimators": 15, "max_depth": 4, "min_samples_leaf": 5, "max_features": 0.5},
              NUMERIC, CATEGORICAL, 1),
    ArenaTask("extra_trees", {"n_estimators": 15, "max_depth": 4, "min_samples_leaf": 5, "max_features": 0.5},
              NUMERIC, CATEGORICAL, 1),
]


def test_sequential_executor_returns_results_in_task_order(data):
    results = run_arena(TASKS, data, 0.05, SequentialExecutor(), log=lambda _m: None)
    assert [r.model for r in results] == [t.model for t in TASKS]
    assert all(r.ok for r in results)


def test_a_failing_task_does_not_stop_the_others(data):
    bad = ArenaTask("logistic_regression", {"not_a_parameter": 1}, NUMERIC, CATEGORICAL, 1)
    results = run_arena([bad, TASKS[1]], data, 0.05, SequentialExecutor(), log=lambda _m: None)
    assert [r.ok for r in results] == [False, True]


def test_one_worker_means_sequential_and_missing_ray_falls_back(monkeypatch):
    assert isinstance(make_executor(1), SequentialExecutor)

    def no_ray(*_args, **_kwargs):
        raise ImportError("no ray")

    messages = []
    monkeypatch.setattr(executor_module, "RayExecutor", no_ray)
    assert isinstance(make_executor(4, log=messages.append), SequentialExecutor)
    assert messages and "not installed" in messages[0]


def test_ray_gives_the_same_scores_as_sequential_and_splits_the_cores(data):
    pytest.importorskip("ray")
    sequential = run_arena(TASKS, data, 0.05, SequentialExecutor(), log=lambda _m: None)
    ray_executor = make_executor(2)
    try:
        assert ray_executor.name == "ray"
        assert ray_executor.threads_per_task == max(1, (os.cpu_count() or 1) // 2)
        parallel = run_arena(TASKS, data, 0.05, ray_executor, log=lambda _m: None)
    finally:
        ray_executor.close()
    assert [r.model for r in parallel] == [t.model for t in TASKS]  # order is preserved
    for a, b in zip(sequential, parallel, strict=True):
        assert b.ok, b.error
        assert b.metrics["pr_auc"] == pytest.approx(a.metrics["pr_auc"], abs=1e-9)


def test_ray_reports_a_task_that_misses_its_deadline_as_failed(data):
    pytest.importorskip("ray")
    ray_executor = make_executor(2, timeout_seconds=0.001)
    try:
        results = run_arena(TASKS[:2], data, 0.05, ray_executor, log=lambda _m: None)
    finally:
        ray_executor.close()
    assert [r.ok for r in results] == [False, False]
    assert all("TimeoutError" in r.error for r in results)
