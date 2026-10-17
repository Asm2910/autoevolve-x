"""Global explanation of the served model: which features drive the risk score?

Run with:  python -m scripts.explain_model

Uses exact tree contributions (log-odds) on the TEST snapshots. Writes
data/processed/feature_importance.csv and explanation_report.md and logs them to
MLflow. Importance = mean |contribution|, also summed per feature group.
"""

from __future__ import annotations

import json

import joblib
import mlflow
import numpy as np
import pandas as pd

from models import tracking
from models.explain import contributions, supports_explanations
from models.training import ARTIFACT_DIR, PROCESSED_DIR, load_config


def main() -> None:
    config = load_config()
    metadata = json.loads((ARTIFACT_DIR / "model_metadata.json").read_text())
    model = joblib.load(ARTIFACT_DIR / metadata["artifact"])
    if not supports_explanations(model):
        raise SystemExit(f"{metadata['model_name']} does not support tree explanations")

    features = metadata["features"]
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    frame = test[features]

    per_feature, bias = contributions(model, frame)
    prob = np.clip(model.predict_proba(frame)[:, 1], 1e-12, 1 - 1e-12)
    logit = np.log(prob / (1 - prob))
    error = float(np.abs(bias + per_feature.sum(axis=1).to_numpy() - logit).max())
    print(f"Additivity check: max |bias + sum(contributions) - logit| = {error:.2e}")
    if error > 1e-3:
        raise SystemExit("explanations are not additive - something is wrong")

    importance = per_feature.abs().mean().sort_values(ascending=False)
    table = pd.DataFrame(
        {
            "feature": importance.index,
            "group": [f.split("__")[0] for f in importance.index],
            "mean_abs_contribution": importance.to_numpy(),
        }
    )
    table["share"] = table["mean_abs_contribution"] / table["mean_abs_contribution"].sum()
    table.to_csv(PROCESSED_DIR / "feature_importance.csv", index=False)
    by_group = table.groupby("group")["share"].sum().sort_values(ascending=False)

    lines = [
        f"# Global explanation ({metadata['model_name']})",
        "",
        f"Test rows: {len(frame):,}. Importance = mean |contribution| in log-odds, as a share of the total.",
        "",
        "## By feature group",
        "",
        "| Group | Share |",
        "|---|---|",
        *[f"| {g} | {s:.1%} |" for g, s in by_group.items()],
        "",
        "## Top 20 features",
        "",
        "| Feature | Group | Share |",
        "|---|---|---|",
        *[f"| {r.feature} | {r.group} | {r.share:.1%} |" for r in table.head(20).itertuples()],
    ]
    (PROCESSED_DIR / "explanation_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[5:]))

    if config.get("tracking", {}).get("enabled", True):
        tracking.setup(config)
        version = tracking.code_version()
        with mlflow.start_run(run_name=f"explain-{version['git_commit']}"):
            mlflow.set_tags({k: str(v) for k, v in version.items()} | {"stage": "explain"})
            mlflow.log_metrics({f"group_share_{g}": float(s) for g, s in by_group.items()})
            mlflow.log_metric("additivity_max_error", error)
            mlflow.log_artifact(str(PROCESSED_DIR / "feature_importance.csv"))
            mlflow.log_artifact(str(PROCESSED_DIR / "explanation_report.md"))
        print("\nLogged to MLflow.")


if __name__ == "__main__":
    main()
