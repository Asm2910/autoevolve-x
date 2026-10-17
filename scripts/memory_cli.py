"""Fill and query the experiment memory.

  python -m scripts.memory_cli ingest            # record the latest training + ablation results
  python -m scripts.memory_cli search "usage features"
  python -m scripts.memory_cli recent
  python -m scripts.memory_cli lessons
  python -m scripts.memory_cli reindex           # rebuild the ChromaDB index from SQLite

Add --embedder hashing to work offline (keyword-like matching, no model download).
Default embedder: minilm (downloads a small model on first use).
"""

from __future__ import annotations

import argparse
import json
import math

import pandas as pd

from memory.embedding import get_embedder
from memory.store import MemoryStore
from memory.vector import VectorIndex
from models.training import ARTIFACT_DIR, PROCESSED_DIR, load_config


def _flat(prefix: str, values: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in values.items() if isinstance(v, int | float)}


def ingest(store: MemoryStore) -> None:
    config = load_config()
    metadata = json.loads((ARTIFACT_DIR / "model_metadata.json").read_text())
    training = metadata["training"]
    common = {
        "dataset_sha256": training["dataset_sha256"],
        "git_commit": training["git_commit"],
        "git_dirty": training.get("git_dirty"),
    }
    features_cfg = config["features"]

    evaluation = json.loads((PROCESSED_DIR / "model_evaluation.json").read_text())
    for name, result in evaluation.items():
        ci = result["test_ci95"]["pr_auc"]
        served = name == metadata["model_name"]
        exp_id, created = store.record_experiment(
            "training",
            name,
            {"model": config["models"][name], "features": features_cfg, "seed": config["training"]["seed"]},
            {**_flat("test", result["test"]), **_flat("val", result["validation"]), "threshold": result["threshold"]},
            hypothesis=f"Baseline training of {name} on all feature groups",
            verdict="served" if served else "not served",
            conclusion=f"{name} test PR-AUC {result['test']['pr_auc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
            **common,
        )
        print(f"training/{name}: {'recorded' if created else 'already known'} (id {exp_id})")

    ablation_path = PROCESSED_DIR / "ablation_results.csv"
    if not ablation_path.exists():
        return
    for row in pd.read_csv(ablation_path).to_dict("records"):
        variant = row["variant"]
        kind, _, group = variant.partition("_")
        metrics = {k: float(v) for k, v in row.items() if isinstance(v, float) and math.isfinite(v)}
        verdict = row.get("verdict") if isinstance(row.get("verdict"), str) else ""
        if variant == "all":
            hypothesis = "Reference: all feature groups"
            conclusion = f"validation PR-AUC {row['val_pr_auc']:.3f}"
        elif kind == "drop":
            hypothesis = f"Does removing the {group} feature group hurt validation PR-AUC?"
            conclusion = (
                f"Dropping {group}: change {row['diff_vs_all']:+.3f} "
                f"[{row['diff_ci_low']:+.3f}, {row['diff_ci_high']:+.3f}] -> {verdict}"
            )
        else:
            hypothesis = f"Is the {group} feature group useful on its own?"
            conclusion = f"{group} alone: validation PR-AUC {row['val_pr_auc']:.3f} ({row['diff_vs_all']:+.3f} vs all)"
        exp_id, created = store.record_experiment(
            "ablation",
            variant,
            {"model": config.get("ablation", {}).get("model", "xgboost"), "groups": row["groups"].split(",")},
            metrics,
            hypothesis=hypothesis,
            verdict=verdict,
            conclusion=conclusion,
            **common,
        )
        if created and kind == "drop":
            if verdict == "group helps":
                store.add_lesson(f"The {group} feature group carries real signal: removing it hurts. {conclusion}",
                                 experiment_id=exp_id, tags=f"ablation {group}")
            elif verdict == "no clear effect":
                store.add_lesson(
                    f"Removing {group} features made no measurable difference; keep or drop is undecided "
                    f"with this data size. {conclusion}",
                    experiment_id=exp_id,
                    tags=f"ablation {group}",
                )
        print(f"ablation/{variant}: {'recorded' if created else 'already known'} (id {exp_id})")


def show(results: list[dict]) -> None:
    for entry in results:
        item = entry["item"]
        via = f" (via {'+'.join(entry['found_by'])})" if "found_by" in entry else ""
        if entry["type"] == "experiments":
            print(f"[{item['kind']}/{item['name']}] {item['verdict'] or '-'}: {item['conclusion']}{via}")
        else:
            print(f"[lesson] {item['text']}{via}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedder", choices=["minilm", "hashing", "none"], default="minilm")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest")
    search = sub.add_parser("search")
    search.add_argument("query")
    sub.add_parser("recent")
    sub.add_parser("lessons")
    sub.add_parser("reindex")
    args = parser.parse_args()

    vectors = None if args.embedder == "none" else VectorIndex(get_embedder(args.embedder))
    store = MemoryStore(vectors=vectors)
    if args.command == "reindex":
        print(f"Reindexed {store.reindex()} documents into ChromaDB ({args.embedder}).")
    elif args.command == "ingest":
        ingest(store)
    elif args.command == "search":
        show(store.search(args.query))
    elif args.command == "recent":
        show([{"type": "experiments", "item": e} for e in store.recent()])
    else:
        show([{"type": "lessons", "item": lesson} for lesson in store.lessons()])


if __name__ == "__main__":
    main()