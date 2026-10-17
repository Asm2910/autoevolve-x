"""Everything the dashboard shows, read from the project's own records. No Streamlit in here."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from memory.store import DEFAULT_PATH, MemoryStore
from models.training import PROCESSED_DIR

AGENT_KIND = "agent_experiment"


def open_store(path: str | Path | None = None) -> MemoryStore:
    return MemoryStore(path or os.environ.get("AUTOEVOLVE_MEMORY_DB") or DEFAULT_PATH)


def latest_final_test(store: MemoryStore) -> dict | None:
    rows = store.recent("final_test", limit=1)
    return rows[0] if rows else None


def agent_ledger(store: MemoryStore, dataset_sha256: str | None = None) -> list[dict]:
    """Every experiment the agent ran on one dataset, oldest first, with its paired difference."""
    rows = store.recent(AGENT_KIND, limit=1000)
    if dataset_sha256 is None and rows:
        dataset_sha256 = rows[0]["dataset_sha256"]
    ledger = []
    for row in reversed(rows):
        if row["dataset_sha256"] != dataset_sha256:
            continue
        m = row["metrics"]
        if "diff_mean" not in m:
            continue
        ledger.append(
            {
                "id": row["id"],
                "hypothesis": row["hypothesis"] or "",
                "verdict": row["verdict"] or "",
                "conclusion": row["conclusion"] or "",
                "mean": float(m["diff_mean"]),
                "low": float(m.get("diff_ci_low", m["diff_mean"])),
                "high": float(m.get("diff_ci_high", m["diff_mean"])),
            }
        )
    return ledger


def lessons(store: MemoryStore, limit: int = 8) -> list[dict]:
    return store.lessons(limit=limit)


def drift_checks(store: MemoryStore) -> list[dict]:
    rows = store.recent("drift_check", limit=200)
    return sorted(rows, key=lambda r: r["name"])


def read_csv(name: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / name
    return pd.read_csv(path) if path.exists() else None


def importance_by_group(table: pd.DataFrame) -> list[tuple[str, float]]:
    shares = table.groupby("group")["share"].sum().sort_values(ascending=False)
    total = float(shares.sum()) or 1.0
    return [(str(group), float(value) / total) for group, value in shares.items()]


def group_of(feature: str) -> str:
    return feature.split("__")[0]


_WINDOW = re.compile(r"^(\d+)d$")


def humanize(feature: str) -> str:
    """usage__max_unique_destinations__30d -> 'Max unique destinations, 30 days'."""
    parts = feature.split("__")
    if len(parts) == 1:
        return feature.replace("_", " ")
    metric = parts[1].replace("_", " ")
    metric = metric[:1].upper() + metric[1:]
    if len(parts) >= 3:
        match = _WINDOW.match(parts[2])
        window = f"{match.group(1)} days" if match else parts[2].replace("_", " ")
        return f"{metric}, {window}"
    return metric


def pretty_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%-d %b %Y")
    except ValueError:
        return iso