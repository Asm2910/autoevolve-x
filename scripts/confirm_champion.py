"""Score the champion on the TEST snapshots, once.

  python -m scripts.confirm_champion

The agent only ever saw train and validation data. This is the single, final look at the test
snapshots. It is refused if a confirmation for this exact dataset is already in memory, because
a second look would turn the test set into one more thing we tuned on.
The result is saved to memory, MLflow (experiment autoevolve-x-final) and a markdown report.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from agents.graph import load_champion
from memory.store import MemoryStore
from models.training import LABEL, PROCESSED_DIR, load_config
from scripts.train_models import file_sha256
from tools.experiment_runner import ExperimentRunner
from tools.final_evaluation import conclusion_text, final_evaluation

KIND = "final_test"


def existing_confirmation(store: MemoryStore, dataset_sha256: str) -> dict | None:
    row = store.db.execute(
        "SELECT id, created_at FROM experiments WHERE kind=? AND dataset_sha256=?", (KIND, dataset_sha256)
    ).fetchone()
    return dict(row) if row else None


def report_markdown(result: dict, conclusion: str) -> str:
    lines = ["# Final test confirmation", "", conclusion, "",
             "| Model | Threshold | PR-AUC | PR-AUC 95% CI | Precision@review | Recall@review |",
             "|---|---|---|---|---|---|"]
    for role in ("champion", "baseline"):
        r = result[role]
        low, high = r["test_ci95"]["pr_auc"]
        lines.append(
            f"| {role} ({r['spec']['model']}) | {r['threshold']:.2f} | {r['test']['pr_auc']:.3f} | "
            f"{low:.3f} to {high:.3f} | {r['test']['precision_at_review']:.3f} | {r['test']['recall_at_review']:.3f} |"
        )
    lines += ["", f"Champion spec: `{json.dumps(result['champion']['spec'])}`", ""]
    return "\n".join(lines)


def log_to_mlflow(config: dict, result: dict, conclusion: str) -> str:
    import mlflow

    from models.tracking import code_version, setup

    base = config.get("tracking", {}).get("experiment_name", "autoevolve-x")
    setup({**config, "tracking": {**config.get("tracking", {}), "experiment_name": f"{base}-final"}})
    with mlflow.start_run(run_name=f"final-test-{result['verdict']}") as run:
        mlflow.set_tags({k: str(v) for k, v in code_version().items()}
                        | {"stage": "final_test", "verdict": result["verdict"], "conclusion": conclusion[:500]})
        mlflow.log_params({"champion_spec": json.dumps(result["champion"]["spec"])[:500]})
        metrics = {"diff_pr_auc_mean": result["diff_pr_auc"]["mean"],
                   "diff_pr_auc_ci_low": result["diff_pr_auc"]["ci95"][0],
                   "diff_pr_auc_ci_high": result["diff_pr_auc"]["ci95"][1]}
        for role in ("champion", "baseline"):
            for key, value in result[role]["test"].items():
                metrics[f"{role}_test_{key}"] = float(value)
            metrics[f"{role}_test_pr_auc_ci_low"] = result[role]["test_ci95"]["pr_auc"][0]
            metrics[f"{role}_test_pr_auc_ci_high"] = result[role]["test_ci95"]["pr_auc"][1]
        mlflow.log_metrics(metrics)
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedder", choices=["minilm", "hashing", "none"], default="minilm")
    args = parser.parse_args()

    config = load_config()
    dataset_sha256 = file_sha256(PROCESSED_DIR / "features.parquet")

    vectors = None
    if args.embedder != "none":
        from memory.embedding import get_embedder
        from memory.vector import VectorIndex

        vectors = VectorIndex(get_embedder(args.embedder))
    store = MemoryStore(vectors=vectors)

    earlier = existing_confirmation(store, dataset_sha256)
    if earlier:
        raise SystemExit(
            f"Refused: the test snapshots were already used to confirm a champion on this dataset "
            f"(memory entry {earlier['id']}, {earlier['created_at']}). Run it once only. "
            f"Use `python -m scripts.memory_cli` to read the earlier result."
        )

    runner = ExperimentRunner(config)  # loads train and validation only
    champion = load_champion(store, runner, dataset_sha256)
    print(f"Champion: {champion.key()}")
    print("Scoring the test snapshots (this is the one and only time) ...")

    manifest = json.loads((PROCESSED_DIR / "feature_manifest.json").read_text())
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    assert LABEL in test.columns
    result = final_evaluation(config, champion, manifest, runner.train, runner.validation, test)
    conclusion = conclusion_text(result)

    from models.tracking import code_version

    version = code_version()
    run_id = ""
    if config.get("tracking", {}).get("enabled", True):
        try:
            run_id = log_to_mlflow(config, result, conclusion)
        except Exception as exc:  # noqa: BLE001 - tracking must not lose the result
            print(f"  (MLflow logging failed: {type(exc).__name__}: {str(exc)[:100]})")

    experiment_id, _ = store.record_experiment(
        KIND, "champion", {"champion": result["champion"]["spec"], "baseline": "logistic_regression"}, result,
        hypothesis="One-time confirmation of the champion on the untouched test snapshots",
        verdict=result["verdict"], conclusion=conclusion, dataset_sha256=dataset_sha256,
        git_commit=str(version.get("git_commit", "")), git_dirty=bool(version.get("git_dirty", False)),
        mlflow_run_id=run_id,
    )
    store.add_lesson(conclusion, experiment_id=experiment_id, tags="final_test")

    report = PROCESSED_DIR / "final_test_report.md"
    report.write_text(report_markdown(result, conclusion))
    print("\n" + conclusion)
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    main()