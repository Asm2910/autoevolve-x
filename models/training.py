"""Training and evaluation building blocks.

Protocol: fit on the training snapshots, choose each model's threshold on the
validation snapshots, then score the test snapshots once with that frozen
threshold and report bootstrap confidence intervals.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from arena import registry

import datasets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = datasets.config_path()  # AUTOEVOLVE_DATASET=ieee switches all of these together
PROCESSED_DIR = datasets.processed_dir()
ARTIFACT_DIR = datasets.artifact_dir()

LABEL = "risk_label"
SEP = "__"
BOOTSTRAP_METRICS = ["pr_auc", "precision", "recall", "f1", "precision_at_review", "recall_at_review"]


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def select_features(manifest: dict, groups: list[str], windows: list[int]) -> list[str]:
    """Pick feature columns by group and window, using the feature naming scheme."""
    allowed_windows = {int(w) for w in windows}
    selected = []
    for column in manifest["features"]:
        parts = column.split(SEP)
        if parts[0] not in groups:
            continue
        if len(parts) == 3:  # <group>__<metric>__<window token>
            numbers = {int(n) for n in re.findall(r"\d+", parts[2])}
            if not numbers <= allowed_windows:
                continue
        selected.append(column)
    return selected


def build_pipeline(name: str, params: dict, numeric: list[str], categorical: list[str], seed: int) -> Pipeline:
    """Preprocessing and model in one artifact. The algorithms live in arena/registry.py."""
    return registry.build_pipeline(name, params, numeric, categorical, seed)


def threshold_grid(config: dict) -> np.ndarray:
    grid = config["training"]["threshold_grid"]
    return np.round(np.arange(grid["start"], grid["stop"] + 1e-9, grid["step"]), 4)


def top_fraction_stats(y: np.ndarray, p: np.ndarray, fraction: float) -> tuple[float, float]:
    """Precision and recall when analysts review only the top-scored fraction."""
    k = max(1, int(np.ceil(fraction * len(y))))
    top = np.argsort(-p, kind="stable")[:k]
    hits = y[top].sum()
    return float(hits / k), float(hits / max(y.sum(), 1))


def sweep_thresholds(y: np.ndarray, p: np.ndarray, grid: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for t in grid:
        pred = (p >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
            }
        )
    table = pd.DataFrame(rows)
    best = float(table.loc[table["f1"].idxmax(), "threshold"])
    return best, table


def point_metrics(y: np.ndarray, p: np.ndarray, threshold: float, review_fraction: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    precision_at_review, recall_at_review = top_fraction_stats(y, p, review_fraction)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision_at_review": precision_at_review,
        "recall_at_review": recall_at_review,
    }


def bootstrap_metrics(
    y: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    review_fraction: float,
    n_samples: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Resample the test rows; every model sees identical resamples (paired)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = {m: {k: [] for k in BOOTSTRAP_METRICS} for m in probabilities}
    for _ in range(n_samples):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0:
            continue
        for name, p in probabilities.items():
            pb = p[idx]
            pred = (pb >= thresholds[name]).astype(int)
            precision_at_review, recall_at_review = top_fraction_stats(yb, pb, review_fraction)
            values = {
                "pr_auc": average_precision_score(yb, pb),
                "precision": precision_score(yb, pred, zero_division=0),
                "recall": recall_score(yb, pred, zero_division=0),
                "f1": f1_score(yb, pred, zero_division=0),
                "precision_at_review": precision_at_review,
                "recall_at_review": recall_at_review,
            }
            for key, value in values.items():
                draws[name][key].append(value)
    return {m: {k: np.array(v) for k, v in d.items()} for m, d in draws.items()}


def interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
