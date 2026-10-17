"""Feature-group ablation: which groups of features actually help?

Run with:  python -m scripts.run_ablation

For one model (default: xgboost) this trains
  - "all"        every configured feature group
  - "drop_<g>"   everything except group g   (does removing g hurt?)
  - "only_<g>"   group g alone               (is g useful by itself?)
Decisions use the VALIDATION snapshots only. Test PR-AUC is printed for reference
and never used to decide anything. Differences are paired bootstrap estimates
(same resampled rows for every variant), so the confidence intervals are honest.
Each variant is logged to MLflow (experiment "<experiment_name>-ablation").
"""

from __future__ import annotations

import copy
import json

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from models import tracking
from models.training import (
    LABEL,
    PROCESSED_DIR,
    build_pipeline,
    interval,
    load_config,
    select_features,
)


def variants(groups: list[str]) -> dict[str, list[str]]:
    out = {"all": list(groups)}
    for group in groups:
        out[f"drop_{group}"] = [g for g in groups if g != group]
    for group in groups:
        out[f"only_{group}"] = [group]
    return out


def verdict(diff_ci: list[float]) -> str:
    """Interpret (drop_g minus all): negative means removing the group hurts."""
    low, high = diff_ci
    if high < 0:
        return "group helps"
    if low > 0:
        return "group hurts"
    return "no clear effect"


def main() -> None:
    config = load_config()
    seed = config["training"]["seed"]
    ablation = config.get("ablation", {})
    model_name = ablation.get("model", "xgboost")
    n_boot = int(ablation.get("bootstrap_samples", 500))
    groups = config["features"]["groups"]
    windows = config["features"]["windows"]

    manifest = json.loads((PROCESSED_DIR / "feature_manifest.json").read_text())
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    validation = pd.read_parquet(PROCESSED_DIR / "validation.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    y_val, y_test = validation[LABEL].to_numpy(), test[LABEL].to_numpy()

    scores: dict[str, dict] = {}
    for name, used in variants(groups).items():
        features = select_features(manifest, used, windows)
        categorical = [c for c in features if c in manifest["categorical"]]
        numeric = [c for c in features if c not in categorical]
        pipeline = build_pipeline(model_name, config["models"][model_name], numeric, categorical, seed)
        pipeline.fit(train[features], train[LABEL])
        val_p = pipeline.predict_proba(validation[features])[:, 1]
        test_p = pipeline.predict_proba(test[features])[:, 1]
        scores[name] = {
            "groups": ",".join(used),
            "n_features": len(features),
            "val_probs": val_p,
            "val_pr_auc": float(average_precision_score(y_val, val_p)),
            "test_pr_auc": float(average_precision_score(y_test, test_p)),
        }
        print(f"{name:22s} features={len(features):3d}  val PR-AUC={scores[name]['val_pr_auc']:.3f}", flush=True)

    # paired bootstrap on validation rows
    rng = np.random.default_rng(seed)
    n = len(y_val)
    draws = {name: np.empty(n_boot) for name in scores}
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_val[idx].sum() == 0:
            idx = rng.integers(0, n, n)
        for name, s in scores.items():
            draws[name][i] = average_precision_score(y_val[idx], s["val_probs"][idx])

    rows = []
    for name, s in scores.items():
        row = {
            "variant": name,
            "groups": s["groups"],
            "n_features": s["n_features"],
            "val_pr_auc": s["val_pr_auc"],
            "val_pr_auc_ci_low": interval(draws[name])[0],
            "val_pr_auc_ci_high": interval(draws[name])[1],
            "test_pr_auc_reference_only": s["test_pr_auc"],
        }
        if name != "all":
            diff = draws[name] - draws["all"]
            row["diff_vs_all"] = float(diff.mean())
            row["diff_ci_low"], row["diff_ci_high"] = interval(diff)
            row["verdict"] = verdict([row["diff_ci_low"], row["diff_ci_high"]]) if name.startswith("drop_") else ""
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(PROCESSED_DIR / "ablation_results.csv", index=False)

    lines = [
        f"# Feature-group ablation ({model_name})",
        "",
        ("Decisions use validation PR-AUC with paired bootstrap 95% CIs. "
         "Test PR-AUC is shown for reference only."),
        "",
        "| Variant | Features | Val PR-AUC [95% CI] | Diff vs all [95% CI] | Verdict | Test PR-AUC (ref) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        diff = (
            f"{r['diff_vs_all']:+.3f} [{r['diff_ci_low']:+.3f}, {r['diff_ci_high']:+.3f}]"
            if "diff_vs_all" in r
            else "-"
        )
        lines.append(
            f"| {r['variant']} | {r['n_features']} | {r['val_pr_auc']:.3f} "
            f"[{r['val_pr_auc_ci_low']:.3f}, {r['val_pr_auc_ci_high']:.3f}] | {diff} | "
            f"{r.get('verdict', '')} | {r['test_pr_auc_reference_only']:.3f} |"
        )
    (PROCESSED_DIR / "ablation_report.md").write_text("\n".join(lines) + "\n")

    if config.get("tracking", {}).get("enabled", True):
        track_cfg = copy.deepcopy(config)
        track_cfg.setdefault("tracking", {})["experiment_name"] = (
            f"{config.get('tracking', {}).get('experiment_name', 'autoevolve-x')}-ablation"
        )
        tracking.setup(track_cfg)
        version = tracking.code_version()
        tags = {k: str(v) for k, v in version.items()} | {"stage": "ablation", "model": model_name}
        with mlflow.start_run(run_name=f"ablation-{version['git_commit']}"):
            mlflow.set_tags(tags)
            mlflow.log_artifact(str(PROCESSED_DIR / "ablation_report.md"))
            mlflow.log_artifact(str(PROCESSED_DIR / "ablation_results.csv"))
            for r in rows:
                with mlflow.start_run(run_name=r["variant"], nested=True):
                    mlflow.set_tags(tags | {"variant": r["variant"], "verdict": r.get("verdict", "")})
                    mlflow.log_params({"groups": r["groups"], "n_features": r["n_features"]})
                    mlflow.log_metrics({k: v for k, v in r.items() if isinstance(v, float)})
        print("\nLogged to MLflow experiment", track_cfg["tracking"]["experiment_name"])

    print("\n" + "\n".join(lines[4:]))


if __name__ == "__main__":
    main()
