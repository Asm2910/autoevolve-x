"""Train every candidate under identical conditions and rank them honestly.

Fair means: same training rows, same features, same validation rows, same bootstrap
resamples for every model. Honest means: a model only ranks above another if the paired
confidence interval says so. Models whose interval overlaps the leader are reported as tied.

The arena only ever reads TRAIN and VALIDATION data. The sealed test snapshots are not
touched here.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from arena.registry import build_pipeline, get_family
from models.training import interval, top_fraction_stats


@dataclass(frozen=True)
class ArenaTask:
    """Everything needed to train one candidate. Plain data, so it can cross a process boundary."""

    model: str
    params: dict
    numeric: list[str]
    categorical: list[str]
    seed: int
    threads: int = -1  # CPU threads this model may use; -1 = all


@dataclass(frozen=True)
class ArenaData:
    """The only data the arena sees: training and validation. The sealed test set is not here."""

    train_x: pd.DataFrame
    train_y: np.ndarray
    val_x: pd.DataFrame
    val_y: np.ndarray


@dataclass
class ArenaResult:
    model: str
    ok: bool
    error: str = ""
    train_seconds: float = 0.0
    predict_seconds: float = 0.0
    size_mb: float = 0.0
    n_features: int = 0
    val_probs: np.ndarray | None = field(default=None, repr=False)
    metrics: dict = field(default_factory=dict)


def build_tasks(
    config: dict, names: list[str], numeric: list[str], categorical: list[str]
) -> list[ArenaTask]:
    seed = config["training"]["seed"]
    missing = [n for n in names if n not in config["models"]]
    if missing:
        raise ValueError(f"No hyperparameters in configs/models.yaml for: {missing}")
    return [ArenaTask(n, dict(config["models"][n]), numeric, categorical, seed) for n in names]


def train_candidate(
    task: ArenaTask,
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    val_x: pd.DataFrame,
    val_y: np.ndarray,
    review_fraction: float,
) -> ArenaResult:
    """Fit one model and score it on validation. Never raises: a failure becomes a result."""
    try:
        pipeline = build_pipeline(
            task.model, task.params, task.numeric, task.categorical, task.seed, task.threads
        )
        limit = task.threads if task.threads > 0 else None  # also caps OpenMP/BLAS pools
        with threadpool_limits(limits=limit):
            started = time.perf_counter()
            pipeline.fit(train_x, train_y)
            train_seconds = time.perf_counter() - started

            started = time.perf_counter()
            probs = pipeline.predict_proba(val_x)[:, 1]
            predict_seconds = time.perf_counter() - started

        buffer = io.BytesIO()
        joblib.dump(pipeline, buffer)
        precision, recall = top_fraction_stats(val_y, probs, review_fraction)
        return ArenaResult(
            model=task.model,
            ok=True,
            train_seconds=train_seconds,
            predict_seconds=predict_seconds,
            size_mb=buffer.tell() / 1e6,
            n_features=len(task.numeric) + len(task.categorical),
            val_probs=probs,
            metrics={
                "pr_auc": float(average_precision_score(val_y, probs)),
                "roc_auc": float(roc_auc_score(val_y, probs)),
                "precision_at_review": precision,
                "recall_at_review": recall,
            },
        )
    except Exception as exc:  # noqa: BLE001 - one broken candidate must not stop the arena
        return ArenaResult(model=task.model, ok=False, error=f"{type(exc).__name__}: {str(exc)[:200]}")


def run_arena(
    tasks: list[ArenaTask],
    data: ArenaData,
    review_fraction: float,
    executor=None,
    log: Callable[[str], None] = print,
) -> list[ArenaResult]:
    """Train every task with the given executor (sequential by default, Ray for parallel)."""
    from arena.executor import SequentialExecutor

    executor = executor or SequentialExecutor()
    return executor.run(tasks, data, review_fraction, log)


def describe(result: ArenaResult) -> str:
    if not result.ok:
        return f"  {result.model:<24} FAILED: {result.error}"
    return (f"  {result.model:<24} PR-AUC {result.metrics['pr_auc']:.3f}  "
            f"train {result.train_seconds:6.1f}s  size {result.size_mb:7.1f} MB")


def paired_pr_auc_draws(
    y: np.ndarray, probs: dict[str, np.ndarray], n_boot: int, seed: int
) -> dict[str, np.ndarray]:
    """Bootstrap PR-AUC. Every model sees the same resampled rows, so differences are paired."""
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = {name: np.empty(n_boot) for name in probs}
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        for name, p in probs.items():
            draws[name][i] = average_precision_score(yb, p[idx])
    return draws


def rank_results(
    results: list[ArenaResult], val_y: np.ndarray, n_boot: int, seed: int
) -> list[dict]:
    """One row per candidate, best first, with confidence intervals and a tie verdict."""
    good = [r for r in results if r.ok]
    draws = paired_pr_auc_draws(val_y, {r.model: r.val_probs for r in good}, n_boot, seed) if good else {}
    ordered = sorted(good, key=lambda r: r.metrics["pr_auc"], reverse=True)
    leader = ordered[0].model if ordered else None

    rows = []
    for rank, r in enumerate(ordered, start=1):
        diff = draws[r.model] - draws[leader]
        ci_low, ci_high = interval(draws[r.model])
        if r.model == leader:
            status = "best"
        elif interval(diff)[1] < 0:
            status = "worse"
        else:
            status = "tied with best"
        rows.append(
            {
                "rank": rank,
                "model": r.model,
                "kind": get_family(r.model).kind,
                "status": status,
                **r.metrics,
                "pr_auc_ci95": [ci_low, ci_high],
                "diff_vs_best": float(diff.mean()),
                "diff_vs_best_ci95": interval(diff),
                "train_seconds": r.train_seconds,
                "predict_seconds": r.predict_seconds,
                "size_mb": r.size_mb,
                "n_features": r.n_features,
            }
        )
    for r in results:
        if not r.ok:
            rows.append({"rank": None, "model": r.model, "kind": get_family(r.model).kind,
                         "status": "failed", "error": r.error})
    return rows


def markdown_report(rows: list[dict], review_fraction: float, n_val: int, n_pos: int) -> str:
    lines = [
        "# Model arena (validation snapshots only)",
        "",
        f"Validation rows: {n_val:,} with {n_pos} positives. The sealed test set was not used. "
        "Intervals are paired bootstrap 95% intervals; \"tied\" means the difference to the leader "
        "is not distinguishable from noise.",
        "",
        f"| # | Model | Kind | PR-AUC [95% CI] | ROC-AUC | P@top{review_fraction:.0%} | "
        f"R@top{review_fraction:.0%} | Train s | Size MB | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["status"] == "failed":
            lines.append(f"| - | {row['model']} | {row['kind']} | failed: {row['error']} | | | | | | failed |")
            continue
        low, high = row["pr_auc_ci95"]
        lines.append(
            f"| {row['rank']} | {row['model']} | {row['kind']} | {row['pr_auc']:.3f} [{low:.3f}, {high:.3f}] | "
            f"{row['roc_auc']:.3f} | {row['precision_at_review']:.3f} | {row['recall_at_review']:.3f} | "
            f"{row['train_seconds']:.1f} | {row['size_mb']:.1f} | {row['status']} |"
        )
    return "\n".join(lines) + "\n"
