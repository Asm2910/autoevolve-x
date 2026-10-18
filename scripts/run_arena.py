"""Train every algorithm on the same data and rank them on the VALIDATION snapshots.

Run with:  python -m scripts.run_arena
Pick some:  python -m scripts.run_arena --models xgboost,lightgbm,catboost

The sealed test snapshots are never read here. Models whose paired confidence interval
overlaps the leader are reported as "tied with best" rather than ranked below it.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from arena.registry import names
from arena.runner import build_tasks, markdown_report, rank_results, run_arena
from models.tracking import code_version, log_arena
from models.training import LABEL, PROCESSED_DIR, load_config, select_features
from scripts.train_models import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", help="comma-separated model names (default: configs/models.yaml arena.models)")
    parser.add_argument("--bootstrap", type=int, help="bootstrap samples (default: arena.bootstrap_samples)")
    parser.add_argument("--no-track", action="store_true", help="do not log to MLflow")
    args = parser.parse_args()

    config = load_config()
    arena_cfg = config.get("arena", {})
    selected = args.models.split(",") if args.models else arena_cfg.get("models") or names()
    unknown = [m for m in selected if m not in names()]
    if unknown:
        parser.error(f"unknown models {unknown}; choose from {names()}")
    n_boot = args.bootstrap or arena_cfg.get("bootstrap_samples", 200)
    review_fraction = config["training"]["review_fraction"]

    manifest = json.loads((PROCESSED_DIR / "feature_manifest.json").read_text())
    features = select_features(manifest, config["features"]["groups"], config["features"]["windows"])
    categorical = [c for c in features if c in manifest["categorical"]]
    numeric = [c for c in features if c not in categorical]

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    validation = pd.read_parquet(PROCESSED_DIR / "validation.parquet")
    y_train, y_val = train[LABEL].to_numpy(), validation[LABEL].to_numpy()

    print(f"Arena: {len(selected)} models, {len(features)} features, "
          f"{len(train):,} training rows, {len(validation):,} validation rows ({int(y_val.sum())} positives)")
    started = time.time()
    tasks = build_tasks(config, selected, numeric, categorical)
    results = run_arena(tasks, train[features], y_train, validation[features], y_val, review_fraction)
    print(f"\nTraining finished in {time.time() - started:.0f}s. Bootstrapping {n_boot} paired resamples ...")

    rows = rank_results(results, y_val, n_boot, config["training"]["seed"])
    report = markdown_report(rows, review_fraction, len(y_val), int(y_val.sum()))
    (PROCESSED_DIR / "arena_validation.json").write_text(json.dumps(rows, indent=2))
    report_path = PROCESSED_DIR / "arena_report.md"
    report_path.write_text(report)
    print("\n" + report)

    if config.get("tracking", {}).get("enabled", True) and not args.no_track:
        run_id = log_arena(
            config,
            rows,
            {"dataset_sha256": file_sha256(PROCESSED_DIR / "features.parquet"), "n_features": len(features)},
            report_path,
        )
        print(f"Logged to MLflow experiment '{config['tracking']['experiment_name']}-arena' (run {run_id}).")
    print(f"Code version: {code_version()['git_commit']}. Wrote {report_path}")


if __name__ == "__main__":
    main()
