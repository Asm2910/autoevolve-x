"""The data swarm: four agents that look at the raw sources and the features before any model does.

  SchemaAgent        Do the tables have the columns, types and categories we expect?
  QualityAgent       Missing values, duplicate ids, constant columns, impossible values, odd timestamps.
  RelationshipAgent  Do the sources join cleanly? Orphans, coverage, hot keys, history before each snapshot.
  FeatureAgent       Which feature groups carry signal once many tests are corrected for? Any leakage?

Like the evaluation swarm these are deterministic agents on the shared harness. The swarm is
generic: it works on any `DataContext` (a dict of tables plus a spec for each), so the same
agents check the synthetic telecom data and the real-data adapter. The feature agent reads the
training split only; the sealed test split is not reachable from here.

Multiple comparisons: with hundreds of features, a few look predictive by chance. The feature agent
turns each feature's p-value into a Benjamini-Hochberg adjusted value and only counts features that
survive at the chosen false discovery rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.stats import chi2_contingency, norm, rankdata

from agentkit.harness import AgentContract, BaseAgent
from swarm.evaluation import AgentFindings, EvaluationSwarm, Finding


# ---- what the agents know about the data ---------------------------------------------------------
@dataclass
class TableSpec:
    required: list[str]
    id_col: str | None = None  # unique row id
    entity_col: str | None = "subscriber_id"  # who the row is about
    time_col: str | None = None
    event_time: bool = True  # False when time_col is a date of record (activation), not an event time
    nonnegative: list[str] = field(default_factory=list)
    categories: dict[str, set] = field(default_factory=dict)


@dataclass
class DataContext:
    tables: dict[str, pd.DataFrame]
    specs: dict[str, TableSpec]
    parent_table: str = "customers"  # the table that lists every entity once
    label_table: str | None = "snapshot_labels"
    label_col: str = "risk_label"
    snapshot_col: str = "snapshot_date"
    entity_col: str = "subscriber_id"
    time_window: tuple[pd.Timestamp, pd.Timestamp] | None = None
    lookback_days: int = 90
    features: pd.DataFrame | None = None  # training split only
    y: np.ndarray | None = None
    categorical: list[str] = field(default_factory=list)
    seed: int = 42


class DataRequest(BaseModel):
    sample_rows: int = Field(default=200_000, ge=1000)
    missing_warn: float = Field(default=0.05, gt=0, lt=1)
    missing_critical: float = Field(default=0.30, gt=0, lt=1)
    fdr: float = Field(default=0.05, gt=0, lt=0.5)  # false discovery rate for the feature tests
    min_effect: float = Field(default=0.03, ge=0, lt=0.5)  # smallest |AUC - 0.5| (or Cramer's V) worth calling signal
    leakage_auc: float = Field(default=0.97, gt=0.5, le=1)
    hot_key_ratio: float = Field(default=100.0, gt=1)


def _contract(name: str, responsibility: str) -> AgentContract:
    return AgentContract(name=name, responsibility=responsibility, kind="deterministic",
                         input_model=DataRequest, output_model=AgentFindings, timeout_seconds=600.0)


class _DataAgent(BaseAgent):
    def __init__(self, context: DataContext, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ctx = context

    def finding(self, severity: str, title: str, detail: str, evidence: dict | None = None, hypothesis: str = "") -> Finding:
        return Finding(agent=self.contract.name, severity=severity, title=title, detail=detail,
                       evidence=evidence or {}, hypothesis=hypothesis)


def _sample(frame: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    return frame if len(frame) <= rows else frame.sample(rows, random_state=seed)


# ---- 1. schema -----------------------------------------------------------------------------------
class SchemaAgent(_DataAgent):
    contract = _contract("data_schema", "Check that every table has the expected columns, types and category values.")

    def execute(self, req: DataRequest) -> AgentFindings:
        c = self.ctx
        findings, summary = [], {}
        for name, spec in c.specs.items():
            frame = c.tables.get(name)
            if frame is None:
                findings.append(self.finding("critical", f"Table {name} is missing", "The table was not loaded."))
                continue
            missing = [col for col in spec.required if col not in frame.columns]
            extra = [col for col in frame.columns if col not in spec.required]
            summary[name] = {"rows": len(frame), "columns": len(frame.columns), "missing_columns": missing}
            if missing:
                findings.append(self.finding("critical", f"{name}: columns missing",
                                             f"Expected columns not found: {', '.join(missing)}.", {"missing": missing}))
            if extra:
                findings.append(self.finding("info", f"{name}: extra columns", f"Not in the expected schema: {', '.join(extra)}."))
            if spec.time_col and spec.time_col in frame.columns and not pd.api.types.is_datetime64_any_dtype(frame[spec.time_col]):
                findings.append(self.finding("warning", f"{name}: {spec.time_col} is not a timestamp",
                                             f"Found type {frame[spec.time_col].dtype}; time windows will be wrong."))
            for col in spec.nonnegative:
                if col in frame.columns and not pd.api.types.is_numeric_dtype(frame[col]):
                    findings.append(self.finding("warning", f"{name}: {col} is not numeric", f"Found type {frame[col].dtype}."))
            for col, allowed in spec.categories.items():
                if col not in frame.columns:
                    continue
                values = frame[col].dropna()
                unknown = values[~values.isin(allowed)]
                if len(unknown):
                    share = len(unknown) / max(len(values), 1)
                    examples = sorted(map(str, unknown.unique()[:5]))
                    findings.append(self.finding(
                        "critical" if share > 0.005 else "warning", f"{name}.{col}: unexpected categories",
                        f"{share:.2%} of values are outside the allowed set, for example {', '.join(examples)}.",
                        {"share": share, "examples": examples}))
        if not any(f.severity in {"critical", "warning"} for f in findings):
            findings.append(self.finding("info", "Schemas match", f"{len(summary)} tables have the expected columns, types and categories."))
        return AgentFindings(agent=self.contract.name, summary=summary, findings=findings)


# ---- 2. quality ----------------------------------------------------------------------------------
class QualityAgent(_DataAgent):
    contract = _contract("data_quality", "Find missing values, duplicates, constants, impossible values and odd timestamps.")

    def execute(self, req: DataRequest) -> AgentFindings:
        c = self.ctx
        findings, summary = [], {}
        for name, spec in c.specs.items():
            frame = c.tables.get(name)
            if frame is None:
                continue
            sample = _sample(frame, req.sample_rows, c.seed)
            info: dict = {"rows": len(frame)}
            if spec.id_col and spec.id_col in frame.columns:
                duplicates = int(frame[spec.id_col].duplicated().sum())
                info["duplicate_ids"] = duplicates
                if duplicates:
                    findings.append(self.finding("critical", f"{name}: duplicate ids",
                                                 f"{duplicates:,} rows repeat an id in {spec.id_col}; counts and sums will be inflated."))
            missing = sample.isna().mean()
            high = missing[missing >= req.missing_warn].sort_values(ascending=False)
            info["columns_with_missing"] = {k: round(float(v), 4) for k, v in high.head(8).items()}
            for col, share in high.items():
                severity = "critical" if share >= req.missing_critical else "warning"
                findings.append(self.finding(severity, f"{name}.{col}: {share:.0%} missing",
                                             f"{share:.1%} of values are empty (sampled).", {"share": float(share)}))
            constant = [col for col in sample.columns if sample[col].nunique(dropna=True) <= 1]
            info["constant_columns"] = constant
            if constant:
                findings.append(self.finding("warning", f"{name}: constant columns",
                                             f"No variation in {', '.join(constant)}; they cannot help a model."))
            for col in spec.nonnegative:
                if col in sample.columns and pd.api.types.is_numeric_dtype(sample[col]):
                    negative = int((sample[col] < 0).sum())
                    if negative:
                        findings.append(self.finding("warning", f"{name}.{col}: negative values",
                                                     f"{negative:,} negative values in a column that should not go below zero."))
            findings += self._timestamps(name, spec, frame)
            summary[name] = info
        if not findings:
            findings.append(self.finding("info", "No quality problems found", "Checked missing values, ids, constants, ranges and timestamps."))
        return AgentFindings(agent=self.contract.name, summary=summary, findings=findings)

    def _timestamps(self, name: str, spec: TableSpec, frame: pd.DataFrame) -> list[Finding]:
        window = self.ctx.time_window
        if not (window and spec.event_time and spec.time_col and spec.time_col in frame.columns
                and pd.api.types.is_datetime64_any_dtype(frame[spec.time_col])):
            return []
        times = frame[spec.time_col]
        outside = int(((times < window[0]) | (times > window[1])).sum())
        if outside:
            return [self.finding("critical", f"{name}: timestamps outside the data window",
                                 f"{outside:,} rows fall outside {window[0]:%Y-%m-%d} to {window[1]:%Y-%m-%d}; "
                                 "events from the future would leak information.")]
        return []


# ---- 3. relationships ----------------------------------------------------------------------------
class RelationshipAgent(_DataAgent):
    contract = _contract("data_relationships", "Check that the sources join cleanly and cover the snapshots.")

    def execute(self, req: DataRequest) -> AgentFindings:
        c = self.ctx
        findings, summary = [], {}
        parent = c.tables.get(c.parent_table)
        if parent is None or c.entity_col not in parent.columns:
            return AgentFindings(agent=self.contract.name, findings=[self.finding(
                "warning", "No parent table", f"{c.parent_table} is needed to check joins.")])
        keys = set(parent[c.entity_col])
        for name, spec in c.specs.items():
            frame = c.tables.get(name)
            if name == c.parent_table or frame is None or c.entity_col not in frame.columns:
                continue
            orphan = float((~frame[c.entity_col].isin(keys)).mean())
            counts = frame[c.entity_col].value_counts()
            coverage = len(keys & set(counts.index)) / max(len(keys), 1)
            median, top = float(counts.median()), int(counts.iloc[0]) if len(counts) else 0
            summary[name] = {"orphan_share": round(orphan, 5), "entity_coverage": round(coverage, 4),
                             "events_per_entity_median": median, "events_per_entity_max": top}
            if orphan > 0:
                findings.append(self.finding("critical" if orphan > 0.01 else "warning", f"{name}: rows without a customer",
                                             f"{orphan:.2%} of rows point at an entity missing from {c.parent_table}; the join would drop them."))
            if coverage < 0.9:
                findings.append(self.finding("warning", f"{name}: low coverage",
                                             f"Only {coverage:.1%} of customers appear in {name}; the rest get empty features.",
                                             hypothesis=f"Coverage of {name} is low, so its feature group may mostly encode 'has activity'."))
            if median > 0 and top > req.hot_key_ratio * median:
                findings.append(self.finding("warning", f"{name}: hot key",
                                             f"One entity has {top:,} rows against a median of {median:g}; it may be a test or shared account."))
            if name != c.label_table:
                findings += self._lookback(name, spec, frame)
        findings += self._labels(parent, keys)
        if not findings:
            findings.append(self.finding("info", "Joins are clean", "Every source joins to the customer table with good coverage."))
        return AgentFindings(agent=self.contract.name, summary=summary, findings=findings)

    def _lookback(self, name: str, spec: TableSpec, frame: pd.DataFrame) -> list[Finding]:
        c = self.ctx
        labels = c.tables.get(c.label_table) if c.label_table else None
        if labels is None or not spec.event_time or not spec.time_col or spec.time_col not in frame.columns or c.snapshot_col not in labels.columns:
            return []
        first_snapshot = pd.to_datetime(labels[c.snapshot_col]).min()
        needed = first_snapshot - pd.Timedelta(days=c.lookback_days)
        earliest = frame[spec.time_col].min()
        if earliest > needed:
            return [self.finding("warning", f"{name}: not enough history before the first snapshot",
                                 f"Events start {earliest:%Y-%m-%d}; the first snapshot ({first_snapshot:%Y-%m-%d}) needs "
                                 f"{c.lookback_days} days back to {needed:%Y-%m-%d}.")]
        return []

    def _labels(self, parent: pd.DataFrame, keys: set) -> list[Finding]:
        c = self.ctx
        labels = c.tables.get(c.label_table) if c.label_table else None
        if labels is None or c.label_col not in labels.columns:
            return []
        out = []
        duplicates = int(labels.duplicated([c.entity_col, c.snapshot_col]).sum())
        if duplicates:
            out.append(self.finding("critical", "Duplicate labels", f"{duplicates:,} (entity, snapshot) pairs have more than one label."))
        orphan = float((~labels[c.entity_col].isin(keys)).mean())
        if orphan:
            out.append(self.finding("warning", "Labels without a customer", f"{orphan:.2%} of labelled rows have no customer record."))
        per_snapshot = labels.groupby(c.snapshot_col)[c.label_col].mean()
        if (per_snapshot == 0).any():
            out.append(self.finding("warning", "Snapshots with no positives",
                                    f"{int((per_snapshot == 0).sum())} snapshot(s) have no positive labels; metrics there are undefined."))
        if len(per_snapshot) > 1 and per_snapshot.min() > 0 and per_snapshot.max() > 3 * per_snapshot.min():
            out.append(self.finding("info", "Positive rate varies across snapshots",
                                    f"From {per_snapshot.min():.2%} to {per_snapshot.max():.2%}; use a time-based split and check drift."))
        return out


# ---- 4. features ---------------------------------------------------------------------------------
def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Adjusted p-values that control the false discovery rate (a feature 'survives' if adjusted <= q)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 1.0
    for rank, idx in zip(range(n, 0, -1), order[::-1], strict=True):
        running = min(running, p[idx] * n / rank)
        adjusted[idx] = running
    return adjusted


def _numeric_test(values: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """AUC of one feature against the label, and a two-sided p-value (rank-sum normal approximation)."""
    v = np.where(np.isnan(values), np.nanmedian(values) if (~np.isnan(values)).any() else 0.0, values)
    n1 = int(y.sum())
    n0 = len(y) - n1
    ranks = rankdata(v)
    auc = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    z = (auc - 0.5) / np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
    return auc, float(2 * norm.sf(abs(z)))


def _categorical_test(values: pd.Series, y: np.ndarray) -> tuple[float, float]:
    """Cramer's V (effect size, 0 to 1) and the chi-square p-value of level against label."""
    table = pd.crosstab(values.astype("object").where(values.notna(), "__missing__"), y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, 1.0
    chi2, p, _, _ = chi2_contingency(table)
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))), float(p)


class FeatureAgent(_DataAgent):
    contract = _contract("data_features", "Rank feature groups by signal after multiple-comparison correction; watch for leakage.")

    def execute(self, req: DataRequest) -> AgentFindings:
        c = self.ctx
        if c.features is None or c.y is None or c.y.sum() < 20:
            return AgentFindings(agent=self.contract.name, findings=[self.finding(
                "info", "No labelled features to test", "Provide the training features and labels to test feature groups.")])
        y = c.y
        rows = []
        for col in c.features.columns:
            if col in c.categorical:
                effect, p = _categorical_test(c.features[col], y)
                auc = float("nan")
            elif pd.api.types.is_numeric_dtype(c.features[col]):
                auc, p = _numeric_test(c.features[col].to_numpy(dtype=float), y)
                effect = abs(auc - 0.5)
            else:
                continue
            rows.append({"feature": col, "group": col.split("__")[0], "window": col.rsplit("__", 1)[-1] if col.rsplit("__", 1)[-1].endswith("d") and "__" in col else "-",
                         "auc": auc, "effect": effect, "p": p})
        table = pd.DataFrame(rows)
        table["p_adjusted"] = benjamini_hochberg(table["p"].to_numpy())
        # a feature counts only if it is statistically real (corrected) AND big enough to matter
        table["survives"] = (table["p_adjusted"] <= req.fdr) & (table["effect"] >= req.min_effect)
        findings = []

        suspects = table[(table["auc"] >= req.leakage_auc) | (table["auc"] <= 1 - req.leakage_auc)]
        for row in suspects.itertuples():
            findings.append(self.finding("critical", f"Possible leakage: {row.feature}",
                                         f"A single feature separates the classes with AUC {row.auc:.3f}. "
                                         "That is rarely real; check that it is not computed from the label or the future.",
                                         {"feature": row.feature, "auc": row.auc}))

        by_group = []
        for group, part in table.groupby("group"):
            tested = len(part)
            kept = int(part["survives"].sum())
            strongest = part.loc[part["effect"].idxmax()]
            by_group.append({"group": group, "tested": tested, "survive": kept, "share": round(kept / tested, 3),
                             "strongest": strongest["feature"], "strongest_effect": round(float(strongest["effect"]), 3)})
        by_group.sort(key=lambda r: (r["share"], r["survive"]), reverse=True)
        total_kept = int(table["survives"].sum())
        summary = {"features_tested": len(table), "survive_fdr": total_kept, "fdr": req.fdr, "groups": by_group,
                   "windows": self._windows(table)}
        top = by_group[0]
        findings.append(self.finding(
            "info", "Signal by feature group",
            f"{total_kept} of {len(table)} features carry signal (false discovery rate {req.fdr:.0%}, effect of at least {req.min_effect}). Strongest group: "
            f"{top['group']} ({top['survive']} of {top['tested']}). "
            + "; ".join(f"{g['group']} {g['survive']}/{g['tested']}" for g in by_group), summary))
        for g in by_group:
            if g["tested"] >= 3 and g["survive"] == 0:
                findings.append(self.finding(
                    "warning", f"No individual signal in {g['group']}",
                    f"None of the {g['tested']} {g['group']} features survives the correction on its own.",
                    {"group": g["group"]},
                    hypothesis=f"The {g['group']} group shows no individual signal after correction; test dropping it, since it may only add noise."))
            elif g is by_group[0] and g["share"] >= 0.5 and g["tested"] >= 3:
                findings.append(self.finding(
                    "info", f"Strong signal in {g['group']}", f"{g['survive']} of {g['tested']} {g['group']} features survive the correction.",
                    {"group": g["group"]},
                    hypothesis=f"The {g['group']} group is rich in signal; test richer {g['group']} features such as more windows or ratios."))
        return AgentFindings(agent=self.contract.name, summary=summary, findings=findings)

    @staticmethod
    def _windows(table: pd.DataFrame) -> list[dict]:
        part = table[table["window"] != "-"]
        out = [{"window": w, "tested": len(g), "survive": int(g["survives"].sum())} for w, g in part.groupby("window")]
        return sorted(out, key=lambda r: r["window"])


AGENTS = (SchemaAgent, QualityAgent, RelationshipAgent, FeatureAgent)


class DataSwarm(EvaluationSwarm):
    agent_classes = AGENTS
    title = "Data swarm report"
    scope = "Raw sources and training features only."
    request_model = DataRequest


# ---- the telecom data set ------------------------------------------------------------------------
def telecom_specs() -> dict[str, TableSpec]:
    from data_generation.validation import CATEGORIES, EXPECTED_COLUMNS, ID_COLUMN

    time_col = {"customers": "activation_date", "snapshot_labels": "snapshot_date"}
    nonneg = {
        "customers": ["tenure_days"],
        "usage_events": ["call_count", "call_duration_sec", "sms_count", "data_mb", "international_calls",
                         "short_calls", "unique_destinations", "night_activity"],
        "recharges": ["recharge_amount"],
        "network_events": ["latency_ms", "packet_loss_pct", "data_session_minutes"],
        "risk_events": ["risk_score", "trigger_count"],
    }
    specs = {}
    for table, columns in EXPECTED_COLUMNS.items():
        specs[table] = TableSpec(
            required=list(columns), id_col=ID_COLUMN.get(table), time_col=time_col.get(table, "timestamp"),
            event_time=table not in time_col or table == "snapshot_labels",
            nonnegative=nonneg.get(table, []),
            categories={col: set(allowed) for (t, col), allowed in CATEGORIES.items() if t == table})
    return specs
