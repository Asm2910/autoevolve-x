import mlflow

from models.tracking import _numeric, code_version, log_training


def test_numeric_keeps_only_finite_numbers():
    out = _numeric({"a": 1, "b": 2.5, "c": float("nan"), "d": "x", "e": float("inf")})
    assert out == {"a": 1.0, "b": 2.5}


def test_code_version_has_dirty_flag():
    version = code_version()
    assert set(version) == {"git_commit", "git_branch", "git_dirty"}
    assert isinstance(version["git_dirty"], bool)


def test_log_training_creates_parent_and_child_runs(tmp_path):
    artifacts = tmp_path / "artifacts"
    processed = tmp_path / "processed"
    artifacts.mkdir()
    processed.mkdir()
    for name in ("model_report.md", "feature_manifest.json"):
        (processed / name).write_text("x")
    (artifacts / "model_metadata.json").write_text("{}")
    (artifacts / "m1.joblib").write_text("x")

    config = {
        "training": {"model_version": "t", "seed": 1},
        "features": {"groups": ["a", "b"], "windows": [7]},
        "models": {"m1": {"depth": 3}},
        "tracking": {
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "artifact_root": str(tmp_path / "mlartifacts"),
            "experiment_name": "test-exp",
        },
    }
    metrics = {"pr_auc": 0.3, "f1": 0.2}
    results = {
        "m1": {
            "threshold": 0.2,
            "test": metrics,
            "validation": metrics,
            "test_ci95": {"pr_auc": [0.2, 0.4]},
        }
    }
    info = {"dataset_sha256": "abc", "n_features": 2, "train_rows": 10, "train_positives": 1}

    import os

    os.environ.pop("MLFLOW_TRACKING_URI", None)
    run_id = log_training(config, results, "m1", info, artifacts, processed)

    runs = mlflow.search_runs(experiment_names=["test-exp"])
    assert len(runs) == 2
    child = runs[runs["tags.mlflow.runName"] == "m1"].iloc[0]
    assert child["metrics.test_pr_auc"] == 0.3
    assert child["metrics.test_pr_auc_ci_low"] == 0.2
    assert child["tags.is_served"] == "True"
    assert child["tags.dataset_sha256"] == "abc"
    assert run_id in set(runs["run_id"])
