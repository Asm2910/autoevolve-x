"""Train, tune and evaluate the candidate models (training.candidates in configs/models.yaml).

Run with:  python -m scripts.train_models
Fewer models (faster):  AUTOEVOLVE_CANDIDATES=logistic_regression,xgboost python -m scripts.train_models

Protocol (no peeking):
  1. fit on the training snapshots
  2. choose each threshold on the validation snapshots
  3. score the test snapshots once with that frozen threshold
  4. bootstrap 95% confidence intervals on the test set
The judge (judge/utility.py) picks the model to serve from validation results alone: utility at the
review cut-off, statistical ties, then cheapest to operate. `training.selection: pr_auc` restores
the old rule (best validation score). The served model is written to model_metadata.json.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from judge.utility import Candidate, UtilityConfig, judge
from models.tracking import code_version, log_training
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
    cost = {}  # training seconds and model size, for the judge

    override = os.environ.get("AUTOEVOLVE_CANDIDATES")  # e.g. CI trains only the fast models
    candidates = (
        [name.strip() for name in override.split(",") if name.strip()]
        if override
        else training.get("candidates") or list(config["models"])
    )
    unknown = [name for name in candidates if name not in config["models"]]
    if unknown:
        raise SystemExit(f"No hyperparameters in configs/models.yaml for: {unknown}")
    print(f"Candidates: {', '.join(candidates)}")
    for name in candidates:
        params = config["models"][name]
        print(f"\nTraining {name} ...")
        pipeline = build_pipeline(name, params, numeric, categorical, seed)
        started = time.perf_counter()
        pipeline.fit(train[features], train[LABEL])
        buffer = io.BytesIO()
        joblib.dump(pipeline, buffer)
        cost[name] = {"train_seconds": time.perf_counter() - started, "size_mb": buffer.tell() / 1e6}

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
    selection = training.get("selection", "utility")
    verdict = None
    if selection == "utility":
        verdict = judge(
            [Candidate(n, val_probs[n], cost[n]["train_seconds"], cost[n]["size_mb"]) for n in results],
            y_val, UtilityConfig.from_config(config), review_fraction, n_boot=200, seed=seed)
        served = verdict.winner
        (PROCESSED_DIR / "judge_report.md").write_text(verdict.markdown())
        (PROCESSED_DIR / "judge_verdict.json").write_text(json.dumps(verdict.to_dict(), indent=2))
        print("\n" + verdict.markdown())
    else:
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
        "selection": {"method": selection, "rationale": verdict.rationale if verdict else f"best validation {metric}"},
        "probability_calibrated": False,
        "features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "metrics": results[served]["test"],
        "validation_metrics": results[served]["validation"],
        "test_ci95": results[served]["test_ci95"],
        "training": {
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **code_version(),
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

    if config.get("tracking", {}).get("enabled", True):
        run_id = log_training(
            config,
            results,
            served,
            {
                "dataset_sha256": metadata["training"]["dataset_sha256"],
                "n_features": len(features),
                "train_rows": len(train),
                "train_positives": int(train[LABEL].sum()),
            },
            ARTIFACT_DIR,
            PROCESSED_DIR,
        )
        print(f"\nLogged to MLflow (run {run_id}). Browse: mlflow ui --backend-store-uri sqlite:///mlflow.db")

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
