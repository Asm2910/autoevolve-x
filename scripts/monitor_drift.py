"""Check whether incoming snapshots have drifted away from the training data.

  python -m scripts.monitor_drift                       # validation snapshots vs the training snapshots
  python -m scripts.monitor_drift --simulate usage:1.6  # rehearsal: scale the usage features of the newest snapshot

Only features are compared (no labels, no model), and only train and validation data are read:
the test snapshots stay untouched. Real checks are recorded in memory; a snapshot that drifted also
leaves a lesson. Simulated checks are NEVER recorded, so a rehearsal cannot pollute what the agent learns.
A 'drift' verdict recommends re-running the agent on fresh data; it does not retrain by itself.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from memory.store import MemoryStore
from models.training import PROCESSED_DIR, load_config
from monitoring.drift import DEFAULTS, feature_drift, simulate_drift, summarize
from scripts.train_models import file_sha256

KIND = "drift_check"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--simulate", default="", help="GROUP:FACTOR, e.g. usage:1.6 (never recorded)")
    parser.add_argument("--embedder", choices=["minilm", "hashing", "none"], default="hashing")
    args = parser.parse_args()

    config = load_config()
    settings = {**DEFAULTS, **config.get("monitoring", {})}
    manifest = json.loads((PROCESSED_DIR / "feature_manifest.json").read_text())
    categorical = list(manifest["categorical"])
    numeric = [c for c in manifest["features"] if c not in categorical]
    dataset_sha256 = file_sha256(PROCESSED_DIR / "features.parquet")

    reference = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    incoming = pd.read_parquet(PROCESSED_DIR / "validation.parquet")  # never the test snapshots
    snapshots = sorted(incoming["snapshot_date"].unique())

    simulated = bool(args.simulate)
    if simulated:
        group, _, factor = args.simulate.partition(":")
        factor = float(factor)
        print(f"SIMULATION: scaling '{group}' features of the newest snapshot by {factor} (nothing is recorded)")

    store = None
    if not simulated:
        vectors = None
        if args.embedder != "none":
            from memory.embedding import get_embedder
            from memory.vector import VectorIndex

            vectors = VectorIndex(get_embedder(args.embedder))
        store = MemoryStore(vectors=vectors)

    print(f"Reference: {reference['snapshot_date'].nunique()} training snapshots, {len(reference)} rows\n")
    table = pd.DataFrame()
    for snapshot in snapshots:
        current = incoming[incoming["snapshot_date"] == snapshot]
        if simulated and snapshot == snapshots[-1]:
            current = simulate_drift(current, group, factor, categorical)
        table = feature_drift(reference, current, numeric, categorical, int(settings["bins"]))
        summary = summarize(table, settings)
        label = str(pd.Timestamp(snapshot).date())
        print(f"{label}  {summary['verdict'].upper():7} {summary['n_major']} major, "
              f"{summary['n_moderate']} moderate of {summary['n_features']} features "
              f"(top: {', '.join(f'{n} {p}' for n, p in summary['top_features'][:3])})")

        if store is not None:
            config_key = {"reference": "train", "snapshot": label, "settings": settings}
            conclusion = (
                f"Snapshot {label}: {summary['verdict']}. {summary['n_major']} of {summary['n_features']} features "
                f"shifted majorly (PSI >= {settings['psi_major']}), {summary['n_moderate']} moderately; "
                f"most shifted: {', '.join(n for n, _ in summary['top_features'][:3])}."
            )
            experiment_id, _ = store.record_experiment(
                KIND, label, config_key, summary, hypothesis="Incoming snapshot resembles the training data",
                verdict=summary["verdict"], conclusion=conclusion, dataset_sha256=dataset_sha256,
            )
            if summary["verdict"] == "drift":
                store.add_lesson(conclusion + " Consider re-running the agent on fresh data.",
                                 experiment_id=experiment_id, tags="drift")

    out = PROCESSED_DIR / ("drift_simulated.csv" if simulated else "drift_latest.csv")
    table.to_csv(out, index=False)
    print(f"\nPer-feature table for the newest snapshot: {out}")
    if simulated:
        print("Rehearsal only. In production a 'drift' verdict on a real snapshot is the signal to re-run the agent.")


if __name__ == "__main__":
    main()