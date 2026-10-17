"""Label-free drift detection: has the incoming data stopped looking like the training data?

PSI (population stability index) compares how rows spread over bins, reference vs current:
  below 0.10  stable        0.10 to 0.25  moderate shift        above 0.25  major shift
The KS statistic (numeric only) is the largest gap between the two cumulative distributions.
No labels and no model are needed, so this can run the day a new snapshot arrives, long before
fraud outcomes are confirmed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-4  # keeps empty bins from producing log(0)

DEFAULTS = {
    "bins": 10,
    "psi_moderate": 0.10,
    "psi_major": 0.25,
    "retrain_share": 0.05,  # retrain when this share of features shows a major shift
}


def _psi_from_counts(reference: np.ndarray, current: np.ndarray) -> float:
    ref = np.maximum(reference / reference.sum(), EPS)
    cur = np.maximum(current / current.sum(), EPS)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Bin edges come from the reference quantiles; missing values get a bin of their own."""
    ref = reference.to_numpy(dtype=float)
    cur = current.to_numpy(dtype=float)
    valid = ref[~np.isnan(ref)]
    if len(valid) == 0 or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(valid, np.linspace(0, 1, bins + 1)[1:-1]))

    def counts(values: np.ndarray) -> np.ndarray:
        out = np.zeros(len(edges) + 2)
        missing = np.isnan(values)
        out[-1] = missing.sum()
        np.add.at(out, np.searchsorted(edges, values[~missing], side="right"), 1)
        return out

    return _psi_from_counts(counts(ref), counts(cur))


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.astype("object").where(reference.notna(), "__missing__").astype(str)
    cur = current.astype("object").where(current.notna(), "__missing__").astype(str)
    levels = sorted(set(ref) | set(cur))
    return _psi_from_counts(
        ref.value_counts().reindex(levels, fill_value=0).to_numpy(dtype=float),
        cur.value_counts().reindex(levels, fill_value=0).to_numpy(dtype=float),
    )


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, numeric: list[str], categorical: list[str], bins: int = 10
) -> pd.DataFrame:
    """One row per feature, most drifted first."""
    rows = []
    for column in numeric:
        ref, cur = reference[column].dropna(), current[column].dropna()
        ks = float(ks_2samp(ref, cur).statistic) if len(ref) > 1 and len(cur) > 1 else 0.0
        rows.append({"feature": column, "kind": "numeric", "psi": numeric_psi(reference[column], current[column], bins),
                     "ks": ks, "missing_ref": float(reference[column].isna().mean()),
                     "missing_cur": float(current[column].isna().mean())})
    for column in categorical:
        rows.append({"feature": column, "kind": "categorical",
                     "psi": categorical_psi(reference[column], current[column]), "ks": float("nan"),
                     "missing_ref": float(reference[column].isna().mean()),
                     "missing_cur": float(current[column].isna().mean())})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def summarize(table: pd.DataFrame, settings: dict | None = None) -> dict:
    """Count shifted features and decide: 'drift' (retraining is worth considering) or 'stable'."""
    cfg = {**DEFAULTS, **(settings or {})}
    major = table["psi"] >= cfg["psi_major"]
    moderate = (table["psi"] >= cfg["psi_moderate"]) & ~major
    share = float(major.mean()) if len(table) else 0.0
    return {
        "n_features": len(table),
        "n_major": int(major.sum()),
        "n_moderate": int(moderate.sum()),
        "share_major": share,
        "top_features": [[row.feature, round(float(row.psi), 4)] for row in table.head(5).itertuples()],
        "verdict": "drift" if share >= cfg["retrain_share"] else "stable",
    }


def simulate_drift(frame: pd.DataFrame, group: str, factor: float, categorical: list[str]) -> pd.DataFrame:
    """Scale every numeric feature of one group, to rehearse what a real shift would look like."""
    out = frame.copy()
    for column in out.columns:
        if column.startswith(f"{group}__") and column not in categorical:
            out[column] = out[column] * factor
    return out