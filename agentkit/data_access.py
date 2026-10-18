"""The agents' window onto the data. The sealed test split is not in it.

Agents and the experiment runner read training and validation snapshots through
`load_split`. Asking for "test" as an agent raises. Only the final evaluation, which runs once
per dataset, passes `role="final_evaluation"`. A separate test scans the agent code to make
sure nothing bypasses this door by reading the test file directly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
AGENT_SPLITS = frozenset({"train", "validation"})
SEALED_SPLITS = frozenset({"test"})


class SealedDataError(PermissionError):
    """An agent asked for data it must never see."""


def load_split(name: str, processed_dir: Path = PROCESSED_DIR, role: str = "agent") -> pd.DataFrame:
    if role == "agent" and name not in AGENT_SPLITS:
        raise SealedDataError(
            f"agents may read {sorted(AGENT_SPLITS)}, not {name!r}. "
            "The test split is scored once, by the final evaluation."
        )
    if role not in {"agent", "final_evaluation"}:
        raise ValueError(f"unknown role {role!r}")
    return pd.read_parquet(processed_dir / f"{name}.parquet")


def assert_agent_readable(path: str | Path) -> None:
    """Refuse any path that points at sealed data (for tools that take file paths)."""
    stem = Path(path).stem.lower()
    if any(sealed in stem for sealed in SEALED_SPLITS):
        raise SealedDataError(f"{path} is sealed data")
