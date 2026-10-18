"""Run the data swarm on the raw sources and the training features.

  python -m scripts.run_data_review

Writes data/processed/data_review_report.{md,json} and memory/data_findings.json (ideas the
proposer reads with `run_agent --use-findings`). Raw tables and the training split only: the
sealed test split is never loaded here.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd
import yaml

import datasets
from agentkit.budget import Budget
from agentkit.data_access import load_split
from agentkit.tracing import Observer, TraceLog
from models.training import LABEL, PROCESSED_DIR, PROJECT_ROOT, load_config
from observability.push import push_at_exit
from swarm.data import DataContext, DataRequest, DataSwarm, ieee_specs, telecom_specs

RAW_DIR = datasets.raw_dir()
FINDINGS_PATH = PROJECT_ROOT / "memory" / "data_findings.json"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    push_at_exit("run_data_review")
    ieee = datasets.active() == "ieee"
    specs = ieee_specs() if ieee else telecom_specs()
    tables = {name: pd.read_parquet(RAW_DIR / f"{name}.parquet") for name in specs}
    if ieee:
        window = None
    else:
        data_cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "data.yaml").read_text())["data"]
        window = (pd.Timestamp(data_cfg["time_range"]["start"]), pd.Timestamp(data_cfg["time_range"]["end"]))

    manifest = json.loads((PROCESSED_DIR / "feature_manifest.json").read_text())
    train = load_split("train", PROCESSED_DIR)
    context = DataContext(
        tables=tables, specs=specs, time_window=window,
        parent_table="transaction" if ieee else "customers", label_table=None if ieee else "snapshot_labels",
        entity_col="TransactionID" if ieee else "subscriber_id", features=train[manifest["features"]],
        y=train[LABEL].to_numpy(), categorical=list(manifest["categorical"]),
        seed=load_config()["training"]["seed"])

    platform = load_config().get("agent_platform", {})
    trace = PROJECT_ROOT / platform.get("trace_dir", "memory/traces") / f"data-review-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    swarm = DataSwarm(context, Observer(TraceLog(trace)), Budget.from_config(platform, PROJECT_ROOT))
    report = swarm.run(DataRequest())
    FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_PATH.write_text(json.dumps(report.hypotheses(), indent=2))
    (PROCESSED_DIR / "data_review_report.md").write_text(report.markdown())
    (PROCESSED_DIR / "data_review_report.json").write_text(json.dumps(report.to_dict(), indent=2, default=str))
    print(report.markdown())
    print(f"Trace: {trace}\nIdeas for the proposer: {FINDINGS_PATH}")


if __name__ == "__main__":
    main()
