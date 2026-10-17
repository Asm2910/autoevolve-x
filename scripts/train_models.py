"""Train, tune and evaluate every model listed in configs/models.yaml.

Run with:  python -m scripts.train_models

Protocol (no peeking):
  1. fit on the training snapshots
  2. choose each threshold on the validation snapshots
  3. score the test snapshots once with that frozen threshold
  4. bootstrap 95% confidence intervals on the test set
The model with the best validation score is written to model_metadata.json and
is the one the API serves.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from models.training import (
    ARTIFACT_DIR,
    LABEL,
    PROCESSED_DIR,
    bootstrap_metrics,
    build_pipeline,
    interval,
    load_config,
    point_metrics,
    select_features,
    sweep_thresholds,
    threshold_grid,
)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    config = load_config()
    training = config["training"]
    seed = training["seed"]
    review_fraction = training["review_fraction"]

    manifest = json.loads((PROCESSED_DIR / "feature_manifest.json").read_text())
    features = select_features(manifest, config["features"]["groups"], config["features"]["windows"])
    categorical = [c for c in features if c in manifest["categorical"]]
    numeric = [c for c in features if c not in categorical]
    print(f"Using {len(features)} features ({len(numeric)} numeric, {len(categorical)} categorical)")

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    validation = pd.read_parquet(PROCESSED_DIR / "validation.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    y_val, y_test = validation[LABEL].to_numpy(), test[LABEL].to_numpy()

    grid = threshold_grid(config)
    fitted, val_probs, test_probs, thresholds, results = {}, {}, {}, {}, {}

    for name, params in config["models"].items():
        print(f"\nTraining {name} ...")
        pipeline = build_pipeline(name, params, numeric, categorical, seed)
        pipeline.fit(train[features], train[LABEL])

        val_probs[name] = pipeline.predict_proba(validation[features])[:, 1]
        test_probs[name] = pipeline.predict_proba(test[features])[:, 1]
        thresholds[name], sweep = sweep_thresholds(y_val, val_probs[name], grid)
        sweep.to_parquet(PROCESSED_DIR / f"{name}_threshold_sweep.parquet", index=False)

        fitted[name] = pipeline
        results[name] = {
            "threshold": thresholds[name],
            "validation": point_metrics(y_val, val_probs[name], thresholds[name], review_fraction),
            "test": point_metrics(y_test, test_probs[name], thresholds[name], review_fraction),
        }
        edge = thresholds[name] in (grid[0], grid[-1])
        print(f"  validation-chosen threshold = {thresholds[name]:.2f}{'  (EDGE OF GRID!)' if edge else ''}")

    print("\nBootstrapping test-set confidence intervals ...")
    draws = bootstrap_metrics(
        y_test, test_probs, thresholds, review_fraction, training["bootstrap_samples"], seed
    )
    for name in results:
        results[name]["test_ci95"] = {k: interval(v) for k, v in draws[name].items()}

    metric = training["selection_metric"]
    served = max(results, key=lambda m: results[m]["validation"][metric])

    for name in results:
        if name == served:
            continue
        results[name]["paired_difference_vs_served"] = {
            k: {
                "mean": float((draws[name][k] - draws[served][k]).mean()),
                "ci95": interval(draws[name][k] - draws[served][k]),
            }
            for k in ("pr_auc", "f1", "precision_at_review")
        }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for name, pipeline in fitted.items():
        joblib.dump(pipeline, ARTIFACT_DIR / f"{name}.joblib")

    metadata = {
        "model_name": served,
        "model_version": training["model_version"],
        "artifact": f"{served}.joblib",
        "threshold": thresholds[served],
        "selection_metric": metric,
        "probability_calibrated": False,
        "features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "metrics": results[served]["test"],
        "validation_metrics": results[served]["validation"],
        "test_ci95": results[served]["test_ci95"],
        "training": {
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "dataset_sha256": file_sha256(PROCESSED_DIR / "features.parquet"),
            "train_rows": len(train),
            "train_positives": int(train[LABEL].sum()),
            "train_snapshots": [str(train["snapshot_date"].min().date()), str(train["snapshot_date"].max().date())],
            "test_snapshots": [str(test["snapshot_date"].min().date()), str(test["snapshot_date"].max().date())],
        },
        "config": config,
    }
    (ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    (PROCESSED_DIR / "model_evaluation.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Model evaluation",
        "",
        f"Test positives: {int(y_test.sum())} of {len(y_test):,} rows. "
        f"Thresholds were chosen on validation and frozen. Served model: **{served}**.",
        "",
        "| Model | Threshold | ROC-AUC | PR-AUC [95% CI] | Precision | Recall | F1 | "
        f"Precision@top{review_fraction:.0%} | Recall@top{review_fraction:.0%} |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        t, ci = r["test"], r["test_ci95"]["pr_auc"]
        lines.append(
            f"| {name} | {r['threshold']:.2f} | {t['roc_auc']:.3f} | {t['pr_auc']:.3f} "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {t['precision']:.3f} | {t['recall']:.3f} | {t['f1']:.3f} | "
            f"{t['precision_at_review']:.3f} | {t['recall_at_review']:.3f} |"
        )
    (PROCESSED_DIR / "model_report.md").write_text("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines[4:]))
    for name in results:
        diff = results[name].get("paired_difference_vs_served")
        if diff:
            pr = diff["pr_auc"]
            print(
                f"\n{name} minus {served}: PR-AUC {pr['mean']:+.3f} "
                f"95% CI [{pr['ci95'][0]:+.3f}, {pr['ci95'][1]:+.3f}]"
                f"{'  -> statistically tied' if pr['ci95'][0] < 0 < pr['ci95'][1] else ''}"
            )
    print(f"\nServed model: {served} (threshold {thresholds[served]:.2f})")
    print(f"Wrote {ARTIFACT_DIR / 'model_metadata.json'}")
    np.seterr(all="warn")


if __name__ == "__main__":
    main()
