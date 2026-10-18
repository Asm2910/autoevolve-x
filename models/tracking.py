"""MLflow experiment tracking for training runs.

Every `scripts.train_models` run becomes one parent MLflow run (the whole pipeline
execution) with one child run per model. Each run records parameters, metrics,
the dataset hash and the exact code version, including whether the working tree
had uncommitted changes, so a result can always be traced back to what produced it.

Browse results with:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "mlartifacts"


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=PROJECT_ROOT
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_version() -> dict[str, str | bool]:
    """Current git commit and whether tracked files have uncommitted changes."""
    commit = _git("rev-parse", "--short", "HEAD") or "unknown"
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": commit,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        # unknown when git is unavailable (e.g. inside a container): treat as dirty
        "git_dirty": True if status is None else bool(status),
    }


def _flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    elif isinstance(value, (list, tuple)):
        out[prefix] = ",".join(str(v) for v in value)[:500]
    else:
        out[prefix] = value


def _numeric(metrics: dict) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)}


def setup(config: dict) -> None:
    """Point MLflow at the local store and select the experiment."""
    tracking = config.get("tracking", {})
    uri = os.environ.get("MLFLOW_TRACKING_URI", tracking.get("tracking_uri") or DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    name = tracking.get("experiment_name", "autoevolve-x")
    if mlflow.get_experiment_by_name(name) is None:
        root = Path(tracking.get("artifact_root") or DEFAULT_ARTIFACT_ROOT)
        mlflow.create_experiment(name, artifact_location=root.resolve().as_uri())
    mlflow.set_experiment(name)


def log_training(
    config: dict,
    results: dict,
    served: str,
    run_info: dict,
    artifact_dir: Path,
    processed_dir: Path,
) -> str:
    """Log one parent run plus a child run per model. Returns the parent run id."""
    setup(config)
    version = code_version()
    tags = {k: str(v) for k, v in version.items()} | {
        "dataset_sha256": run_info["dataset_sha256"],
        "model_version": str(config["training"]["model_version"]),
        "served_model": served,
        "stage": "training",
    }

    with mlflow.start_run(run_name=f"train-{version['git_commit']}") as parent:
        mlflow.set_tags(tags)
        params: dict = {}
        _flatten("training", config["training"], params)
        _flatten("features", config["features"], params)
        params.update(
            n_features=run_info["n_features"],
            train_rows=run_info["train_rows"],
            train_positives=run_info["train_positives"],
        )
        mlflow.log_params(params)
        mlflow.log_artifact(str(processed_dir / "model_report.md"))
        mlflow.log_artifact(str(processed_dir / "feature_manifest.json"))
        mlflow.log_artifact(str(artifact_dir / "model_metadata.json"))

        for name, result in results.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tags(tags | {"model": name, "is_served": str(name == served)})
                model_params: dict = {}
                _flatten("model", config["models"][name], model_params)
                mlflow.log_params(model_params | {"threshold": result["threshold"]})
                metrics = {f"test_{k}": v for k, v in _numeric(result["test"]).items()}
                metrics |= {f"val_{k}": v for k, v in _numeric(result["validation"]).items()}
                for key, (low, high) in result["test_ci95"].items():
                    metrics[f"test_{key}_ci_low"] = float(low)
                    metrics[f"test_{key}_ci_high"] = float(high)
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(artifact_dir / f"{name}.joblib"))
                sweep = processed_dir / f"{name}_threshold_sweep.parquet"
                if sweep.exists():
                    mlflow.log_artifact(str(sweep))
        return parent.info.run_id



def log_agent_candidate(
    config: dict, identity: dict, hypothesis: str, result: dict, verdict: str, reason: str
) -> str:
    """One MLflow run per experiment the agent tried (experiment '<name>-agent')."""
    agent_config = {
        **config,
        "tracking": {
            **config.get("tracking", {}),
            "experiment_name": f"{config.get('tracking', {}).get('experiment_name', 'autoevolve-x')}-agent",
        },
    }
    setup(agent_config)
    version = code_version()
    tags = {k: str(v) for k, v in version.items()} | {
        "stage": "agent",
        "verdict": verdict,
        "reason": reason[:500],
        "hypothesis": hypothesis[:500],
    }
    with mlflow.start_run(run_name=f"{identity['model']}-{verdict}") as run:
        mlflow.set_tags(tags)
        params: dict = {}
        _flatten("spec", identity, params)
        mlflow.log_params(params)
        mlflow.log_metrics(
            _numeric(result)
            | {"diff_ci_low": float(result["diff_ci"][0]), "diff_ci_high": float(result["diff_ci"][1])}
        )
        return run.info.run_id


def log_arena(config: dict, rows: list[dict], run_info: dict, report_path: Path) -> str:
    """One parent run for the arena plus a child run per candidate ('<name>-arena' experiment)."""
    arena_config = {
        **config,
        "tracking": {
            **config.get("tracking", {}),
            "experiment_name": f"{config.get('tracking', {}).get('experiment_name', 'autoevolve-x')}-arena",
        },
    }
    setup(arena_config)
    version = code_version()
    tags = {k: str(v) for k, v in version.items()} | {
        "stage": "arena",
        "dataset_sha256": run_info["dataset_sha256"],
    }
    with mlflow.start_run(run_name=f"arena-{version['git_commit']}") as parent:
        mlflow.set_tags(tags)
        mlflow.log_params({"n_features": run_info["n_features"], "n_models": len(rows)})
        mlflow.log_artifact(str(report_path))
        for row in rows:
            with mlflow.start_run(run_name=row["model"], nested=True):
                mlflow.set_tags(tags | {"model": row["model"], "kind": row["kind"], "status": row["status"]})
                if row["status"] == "failed":
                    mlflow.set_tag("error", row["error"][:500])
                    continue
                model_params: dict = {}
                _flatten("model", config["models"][row["model"]], model_params)
                mlflow.log_params(model_params)
                metrics = _numeric({k: row[k] for k in (
                    "pr_auc", "roc_auc", "precision_at_review", "recall_at_review",
                    "train_seconds", "predict_seconds", "size_mb", "diff_vs_best")})
                metrics["pr_auc_ci_low"], metrics["pr_auc_ci_high"] = row["pr_auc_ci95"]
                mlflow.log_metrics(metrics)
        return parent.info.run_id
