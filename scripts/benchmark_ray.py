"""Measure what Ray buys: the same arena at 1, 2 and 4 workers.

Run with:  python -m scripts.benchmark_ray
Choose:    python -m scripts.benchmark_ray --workers 1,2,4 --models xgboost,lightgbm,catboost,random_forest

Worker count 1 is the sequential baseline (one model at a time, using every core). With more
workers, several models train at once and each gets its share of the cores. Speed-up is limited
by the slowest single model and by how many cores the machine really has, so on a 2-core laptop
expect roughly nothing. The benchmark also checks that parallel runs give the same scores.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

from arena.executor import make_executor
from arena.runner import ArenaData, build_tasks, run_arena
from models.training import LABEL, PROCESSED_DIR, load_config, select_features

DEFAULT_MODELS = [
    "logistic_regression", "random_forest", "extra_trees", "xgboost",
    "lightgbm", "hist_gradient_boosting", "catboost",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", default="1,2,4", help="comma-separated worker counts")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = parser.parse_args()

    config = load_config()
    counts = [int(w) for w in args.workers.split(",")]
    models = args.models.split(",")
    manifest = json.loads((PROCESSED_DIR / "feature_manifest.json").read_text())
    features = select_features(manifest, config["features"]["groups"], config["features"]["windows"])
    categorical = [c for c in features if c in manifest["categorical"]]
    numeric = [c for c in features if c not in categorical]
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    validation = pd.read_parquet(PROCESSED_DIR / "validation.parquet")
    data = ArenaData(train[features], train[LABEL].to_numpy(), validation[features], validation[LABEL].to_numpy())
    tasks = build_tasks(config, models, numeric, categorical)
    timeout = config.get("arena", {}).get("task_timeout_seconds")

    cores = os.cpu_count() or 1
    print(f"Benchmark: {len(models)} models, {cores} CPU cores, worker counts {counts}\n")
    rows, baseline_scores = [], None
    for workers in counts:
        executor = make_executor(workers, timeout)
        started = time.perf_counter()
        try:
            results = run_arena(tasks, data, config["training"]["review_fraction"], executor, log=lambda _m: None)
        finally:
            executor.close()
        wall = time.perf_counter() - started
        scores = {r.model: r.metrics["pr_auc"] for r in results if r.ok}
        if baseline_scores is None:
            baseline_scores = scores
        drift = max((abs(scores[m] - baseline_scores[m]) for m in scores if m in baseline_scores), default=0.0)
        rows.append(
            {
                "workers": workers,
                "executor": executor.name,
                "wall_seconds": wall,
                "sum_of_model_seconds": sum(r.train_seconds for r in results if r.ok),
                "failed": [r.model for r in results if not r.ok],
                "max_pr_auc_difference_vs_first_run": drift,
            }
        )
        print(f"  {workers} worker(s) [{executor.name}]: {wall:6.1f}s wall clock, max PR-AUC difference {drift:.4f}")

    base = rows[0]["wall_seconds"]
    for row in rows:
        row["speedup"] = base / row["wall_seconds"]
        row["efficiency"] = row["speedup"] / row["workers"]

    lines = [
        "# Ray benchmark", "",
        f"{len(models)} models, {cores} CPU cores. Speed-up is relative to the first row.", "",
        "| Workers | Executor | Wall clock (s) | Speed-up | Efficiency | Max PR-AUC difference |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['workers']} | {r['executor']} | {r['wall_seconds']:.1f} | {r['speedup']:.2f}x | "
            f"{r['efficiency']:.0%} | {r['max_pr_auc_difference_vs_first_run']:.4f} |"
        )
    (PROCESSED_DIR / "ray_benchmark.json").write_text(json.dumps({"cores": cores, "models": models, "rows": rows}, indent=2))
    (PROCESSED_DIR / "ray_benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {PROCESSED_DIR / 'ray_benchmark.md'}")


if __name__ == "__main__":
    main()
