"""The one-time, honest score of the champion on the TEST snapshots.

Protocol (same as the normal training run, no peeking):
  1. fit the champion and the logistic-regression baseline on the TRAIN snapshots
  2. choose each model's decision threshold on the VALIDATION snapshots
  3. score the TEST snapshots once with those frozen thresholds
  4. bootstrap 95% intervals, including the paired PR-AUC difference (champion minus baseline)

This module only computes. Whether it may run at all (once per dataset) is decided by the script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.spec import ExperimentSpec
from models.training import (
    LABEL,
    bootstrap_metrics,
    build_pipeline,
    interval,
    point_metrics,
    select_features,
    sweep_thresholds,
    threshold_grid,
)


def _fit(spec: ExperimentSpec, config: dict, manifest: dict, train: pd.DataFrame) -> tuple:
    features = select_features(manifest, spec.groups, spec.windows)
    categorical = [c for c in features if c in manifest["categorical"]]
    numeric = [c for c in features if c not in categorical]
    params = spec.resolved_params(config["models"][spec.model])
    pipeline = build_pipeline(spec.model, params, numeric, categorical, config["training"]["seed"])
    pipeline.fit(train[features], train[LABEL])
    return pipeline, features


def final_evaluation(
    config: dict,
    champion: ExperimentSpec,
    manifest: dict,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    training = config["training"]
    baseline = ExperimentSpec(
        model="logistic_regression",
        groups=list(config["features"]["groups"]),
        windows=list(config["features"]["windows"]),
        hypothesis="Simple baseline: logistic regression on all features",
    )
    specs = {"champion": champion, "baseline": baseline}
    y_val, y_test = validation[LABEL].to_numpy(), test[LABEL].to_numpy()
    grid = threshold_grid(config)

    thresholds: dict[str, float] = {}
    test_probs: dict[str, np.ndarray] = {}
    for role, spec in specs.items():
        pipeline, features = _fit(spec, config, manifest, train)
        thresholds[role], _ = sweep_thresholds(y_val, pipeline.predict_proba(validation[features])[:, 1], grid)
        test_probs[role] = pipeline.predict_proba(test[features])[:, 1]  # the only look at the test data

    draws = bootstrap_metrics(
        y_test, test_probs, thresholds, training["review_fraction"],
        training["bootstrap_samples"], training["seed"],
    )
    diff = draws["champion"]["pr_auc"] - draws["baseline"]["pr_auc"]  # paired: same resampled rows

    result: dict = {}
    for role, spec in specs.items():
        result[role] = {
            "spec": spec.identity(),
            "threshold": thresholds[role],
            "test": point_metrics(y_test, test_probs[role], thresholds[role], training["review_fraction"]),
            "test_ci95": {k: interval(v) for k, v in draws[role].items()},
        }
    ci = interval(diff)
    result["diff_pr_auc"] = {"mean": float(diff.mean()), "ci95": ci}
    result["n_test_rows"] = len(y_test)
    result["n_test_positives"] = int(y_test.sum())
    result["verdict"] = "confirmed" if ci[0] > 0 else "not_confirmed"
    return result


def conclusion_text(result: dict) -> str:
    champ, base = result["champion"], result["baseline"]
    low, high = result["diff_pr_auc"]["ci95"]
    ch_low, ch_high = champ["test_ci95"]["pr_auc"]
    verdict = (
        "The champion is clearly better than the logistic-regression baseline."
        if result["verdict"] == "confirmed"
        else "The champion is NOT clearly better than the logistic-regression baseline: the interval includes zero."
    )
    return (
        f"Final test confirmation on {result['n_test_rows']} rows ({result['n_test_positives']} fraud). "
        f"Champion {champ['spec']['model']} test PR-AUC {champ['test']['pr_auc']:.3f} "
        f"(95% CI {ch_low:.3f} to {ch_high:.3f}); baseline {base['test']['pr_auc']:.3f}. "
        f"Paired difference {result['diff_pr_auc']['mean']:+.3f} (95% CI {low:+.3f} to {high:+.3f}). {verdict}"
    )