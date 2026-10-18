"""Which dataset the project is working on.

Default is the synthetic telecom data. Setting the environment variable
AUTOEVOLVE_DATASET=ieee switches every script to the IEEE-CIS real-data copy: its own processed
folder, its own configuration, its own model folder and its own agent memory. Nothing from one
dataset can overwrite the other, and the served telecom model is never touched.

    AUTOEVOLVE_DATASET=ieee python -m scripts.run_arena
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KNOWN = ("telecom", "ieee")


def active() -> str:
    name = os.environ.get("AUTOEVOLVE_DATASET", "telecom").strip().lower() or "telecom"
    if name not in KNOWN:
        raise SystemExit(f"AUTOEVOLVE_DATASET must be one of {KNOWN}, got {name!r}")
    return name


def processed_dir() -> Path:
    return ROOT / "data" / ("processed" if active() == "telecom" else f"{active()}/processed")


def raw_dir() -> Path:
    return ROOT / "data" / ("raw" if active() == "telecom" else f"{active()}/raw")


def config_path() -> Path:
    return ROOT / "configs" / ("models.yaml" if active() == "telecom" else f"{active()}.yaml")


def artifact_dir() -> Path:
    return ROOT / "models" / ("artifacts" if active() == "telecom" else f"artifacts_{active()}")


def memory_suffix() -> str:
    return "" if active() == "telecom" else f"_{active()}"
