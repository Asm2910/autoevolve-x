"""The evaluation swarm: four small agents that each look at a model from a different angle.

  PerformanceAgent  How good is it overall, and is it steady over time?
  SegmentAgent      Is there a customer group where it quietly does worse?
  ErrorAgent        How do the frauds it misses differ from the ones it catches?
  DriftAgent        Has the data it is scoring stopped looking like the data it learned from?

All four are deterministic code (kind="deterministic"). The value of a swarm here is not
"more LLM calls": each agent has one question, a typed contract, a time limit and a trace
record, and a failure in one never hides the findings of the others. The swarm merges their
findings into one ranked report, and turns the warnings into hypotheses the proposer agent
can read as evidence (never as instructions: they go through the guardrails like any outside text).

Data rules: agents see validation scores and the train snapshots only. The sealed test split
is not reachable from this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import average_precision_score, roc_auc_score

from agentkit.budget import Budget
from agentkit.harness import AgentContract, BaseAgent
from agentkit.tracing import Observer
from monitoring.drift import feature_drift, summarize
from observability import agent_metrics as metrics

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


# ---- shared memory and the typed messages --------------------------------------------------------
@dataclass
class EvalContext:
    """What every evaluation agent can read. Shared and read-only by convention."""

    reference: pd.DataFrame  # training snapshots (what the model learned from)
    frame: pd.DataFrame  # validation snapshots (what is being judged)
    y: np.ndarray
    scores: np.ndarray
    numeric: list[str]
    categorical: list[str]
    review_fraction: float = 0.01
    seed: int = 42
    dates: pd.Series | None = None
    drift_settings: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (len(self.frame) == len(self.y) == len(self.scores)):
            raise ValueError("frame, labels and scores must have the same length")

    @property
    def cutoff(self) -> float:
        """The score above which a case goes to a reviewer (the top review_fraction)."""
        return float(np.quantile(self.scores, 1 - self.review_fraction))


class EvalRequest(BaseModel):
    n_boot: int = Field(default=200, ge=20, le=5000)
    min_segment_rows: int = Field(default=500, ge=1)
    min_segment_positives: int = Field(default=15, ge=1)
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    top_n: int = Field(default=5, ge=1, le=20)
    time_buckets: int = Field(default=4, ge=2, le=12)


class Finding(BaseModel):
    agent: str
    severity: str  # "critical" | "warning" | "info"
    title: str
    detail: str
    evidence: dict = Field(default_factory=dict)
    hypothesis: str = ""  # what an experiment could try, if the finding is a warning


class AgentFindings(BaseModel):
    agent: str
    summary: dict = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)


def _contract(name: str, responsibility: str, timeout: float = 300.0) -> AgentContract:
    return AgentContract(
        name=name,
        responsibility=responsibility,
        kind="deterministic",
        input_model=EvalRequest,
        output_model=AgentFindings,
        timeout_seconds=timeout,
    )


class _EvalAgent(BaseAgent):
    def __init__(self, context: EvalContext, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ctx = context


def _safe_pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


# ---- 1. performance ------------------------------------------------------------------------------
class PerformanceAgent(_EvalAgent):
    contract = _contract("eval_performance", "Measure overall quality with an interval, and check it holds over time.")

    def execute(self, req: EvalRequest) -> AgentFindings:
        c = self.ctx
        rng = np.random.default_rng(c.seed)
        n, base_rate = len(c.y), float(c.y.mean())
        pr_auc = _safe_pr_auc(c.y, c.scores)
        roc = float(roc_auc_score(c.y, c.scores)) if 0 < c.y.sum() < n else float("nan")
        draws = []
        for _ in range(req.n_boot):
            idx = rng.integers(0, n, n)
            draws.append(_safe_pr_auc(c.y[idx], c.scores[idx]))
        low, high = (float(v) for v in np.nanpercentile(draws, [2.5, 97.5]))

        flagged = c.scores >= c.cutoff
        precision = float(c.y[flagged].mean()) if flagged.any() else 0.0
        recall = float(flagged[c.y == 1].mean()) if c.y.sum() else 0.0
        lift = precision / base_rate if base_rate else float("nan")
        summary = {
            "rows": n, "positives": int(c.y.sum()), "base_rate": round(base_rate, 5),
            "pr_auc": round(pr_auc, 4), "pr_auc_ci": [round(low, 4), round(high, 4)], "roc_auc": round(roc, 4),
            "precision_at_review": round(precision, 4), "recall_at_review": round(recall, 4), "lift": round(lift, 2),
        }
        findings = [Finding(
            agent=self.contract.name, severity="info", title="Overall quality",
            detail=(f"PR-AUC {pr_auc:.3f} (95% interval {low:.3f} to {high:.3f}) against a base rate of {base_rate:.3%}. "
                    f"Reviewing the top {c.review_fraction:.1%} finds {recall:.1%} of fraud at {precision:.1%} precision "
                    f"({lift:.1f}x lift)."),
            evidence=summary)]
        if low < 2 * base_rate:
            findings.append(Finding(
                agent=self.contract.name, severity="warning", title="Weak signal",
                detail=f"The lower end of the PR-AUC interval ({low:.3f}) is under twice the base rate ({base_rate:.3%}).",
                evidence={"pr_auc_low": low, "base_rate": base_rate},
                hypothesis="Add stronger feature groups or a different model family; the current signal is close to chance."))
        findings += self._stability(req, pr_auc)
        return AgentFindings(agent=self.contract.name, summary=summary, findings=findings)

    def _stability(self, req: EvalRequest, overall: float) -> list[Finding]:
        c = self.ctx
        if c.dates is None:
            return []
        order = pd.to_datetime(c.dates).rank(method="first").to_numpy()
        buckets = pd.qcut(order, req.time_buckets, labels=False, duplicates="drop")
        scores = []
        for b in np.unique(buckets):
            mask = buckets == b
            if c.y[mask].sum() >= 5:
                scores.append(_safe_pr_auc(c.y[mask], c.scores[mask]))
        if len(scores) < 2:
            return []
        worst = float(np.nanmin(scores))
        evidence = {"by_period": [round(float(s), 4) for s in scores], "overall": round(overall, 4)}
        if worst < 0.6 * overall:
            return [Finding(
                agent=self.contract.name, severity="warning", title="Quality is not steady over time",
                detail=f"PR-AUC by period runs from {worst:.3f} to {max(scores):.3f} against {overall:.3f} overall.",
                evidence=evidence,
                hypothesis="Behaviour features over shorter windows (7 or 30 days) may track recent change better.")]
        return [Finding(agent=self.contract.name, severity="info", title="Quality is steady over time",
                        detail=f"PR-AUC by period stays between {worst:.3f} and {max(scores):.3f}.", evidence=evidence)]


# ---- 2. segments ---------------------------------------------------------------------------------
class SegmentAgent(_EvalAgent):
    contract = _contract("eval_segments", "Find customer groups where fraud is caught less often than overall.")

    def execute(self, req: EvalRequest) -> AgentFindings:
        c = self.ctx
        rng = np.random.default_rng(c.seed + 1)
        caught = (c.scores >= c.cutoff).astype(float)
        overall_recall = float(caught[c.y == 1].mean()) if c.y.sum() else float("nan")
        rows = []
        for column in c.categorical:
            values = c.frame[column].astype("object").where(c.frame[column].notna(), "__missing__").astype(str).to_numpy()
            for level in np.unique(values):
                mask = values == level
                positives = int(c.y[mask].sum())
                if mask.sum() < req.min_segment_rows or positives < req.min_segment_positives:
                    continue
                rows.append((column, level, mask, positives))
        tests = max(len(rows), 1)
        alpha = req.alpha / tests  # Bonferroni: many groups are tested, so demand stronger evidence
        table, flagged = [], []
        for column, level, mask, positives in rows:
            hits = caught[mask & (c.y == 1)]
            recall = float(hits.mean())
            draws = hits[rng.integers(0, len(hits), (req.n_boot, len(hits)))].mean(axis=1)
            upper = float(np.quantile(draws, 1 - alpha / 2))
            table.append({"segment": f"{column}={level}", "rows": int(mask.sum()), "positives": positives,
                          "recall": round(recall, 4), "recall_upper": round(upper, 4)})
            if upper < overall_recall:
                short = column.split("__")[-1]
                flagged.append(((overall_recall - recall) * positives, Finding(
                    agent=self.contract.name, severity="warning",
                    title=f"Weaker on {short}={level}",
                    detail=(f"At the review cut-off the model catches {recall:.1%} of fraud in {short}={level} "
                            f"({positives} cases) against {overall_recall:.1%} overall; even the upper end "
                            f"({upper:.1%}, adjusted for {tests} groups tested) stays below."),
                    evidence=table[-1] | {"overall_recall": round(overall_recall, 4), "groups_tested": tests},
                    hypothesis=f"Fraud in {short}={level} is caught less often; try features or settings that "
                               f"help that group (for example another feature group or a deeper model).")))
        flagged.sort(key=lambda item: item[0], reverse=True)  # biggest shortfall in fraud cases first
        findings = [finding for _, finding in flagged[: req.top_n]]
        if len(flagged) > req.top_n:
            rest = ", ".join(f.title.removeprefix("Weaker on ") for _, f in flagged[req.top_n: req.top_n + 8])
            findings.append(Finding(agent=self.contract.name, severity="info",
                                    title=f"{len(flagged) - req.top_n} more groups are also weaker",
                                    detail=f"Smaller shortfalls: {rest}{'...' if len(flagged) > req.top_n + 8 else ''}."))
        table.sort(key=lambda r: r["recall"])
        if not findings:
            findings.append(Finding(
                agent=self.contract.name, severity="info", title="No group is clearly underserved",
                detail=f"{len(table)} groups tested at the {alpha:.4f} level (corrected for multiple tests); none is clearly below the overall recall of {overall_recall:.1%}.",
                evidence={"groups_tested": len(table)}))
        return AgentFindings(agent=self.contract.name,
                             summary={"groups_tested": len(table), "overall_recall": round(overall_recall, 4),
                                      "weakest": table[: req.top_n]},
                             findings=findings)


# ---- 3. errors -----------------------------------------------------------------------------------
class ErrorAgent(_EvalAgent):
    contract = _contract("eval_errors", "Explain what separates the frauds the model misses from the ones it catches.")

    def execute(self, req: EvalRequest) -> AgentFindings:
        c = self.ctx
        flagged = c.scores >= c.cutoff
        fraud = c.y == 1
        missed, caught = fraud & ~flagged, fraud & flagged
        summary = {"missed": int(missed.sum()), "caught": int(caught.sum()),
                   "false_alarms": int((flagged & ~fraud).sum())}
        if missed.sum() < 10 or caught.sum() < 10:
            return AgentFindings(agent=self.contract.name, summary=summary, findings=[Finding(
                agent=self.contract.name, severity="info", title="Too few errors to analyse",
                detail=f"{summary['missed']} missed and {summary['caught']} caught frauds; at least 10 of each are needed.")])
        effects = []
        for column in c.numeric:
            a = c.frame.loc[missed, column].to_numpy(dtype=float)
            b = c.frame.loc[caught, column].to_numpy(dtype=float)
            a, b = a[~np.isnan(a)], b[~np.isnan(b)]
            if len(a) < 5 or len(b) < 5:
                continue
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            if pooled > 0:
                effects.append((column, float((a.mean() - b.mean()) / pooled)))
        effects.sort(key=lambda item: abs(item[1]), reverse=True)
        top = effects[: req.top_n]
        groups: dict[str, int] = {}
        for column, _ in effects[:20]:
            groups[column.split("__")[0]] = groups.get(column.split("__")[0], 0) + 1
        lead_group = max(groups, key=groups.get) if groups else ""
        summary["top_differences"] = [[name, round(d, 2)] for name, d in top]
        summary["group_counts_in_top20"] = groups
        findings = [Finding(
            agent=self.contract.name, severity="info", title="What the misses look like",
            detail=(f"{summary['missed']} of {int(fraud.sum())} frauds fall below the review cut-off. They differ most from "
                    f"the caught frauds on: " + ", ".join(f"{n} ({d:+.2f} sd)" for n, d in top) + "."),
            evidence=summary)]
        if lead_group and top and abs(top[0][1]) >= 0.5:
            findings.append(Finding(
                agent=self.contract.name, severity="warning", title=f"Misses concentrate in {lead_group} features",
                detail=f"{groups[lead_group]} of the 20 features that separate missed from caught fraud belong to the {lead_group} group.",
                evidence={"groups": groups},
                hypothesis=f"Missed fraud differs mainly in {lead_group} features; try other windows or interactions for that group."))
        return AgentFindings(agent=self.contract.name, summary=summary, findings=findings)


# ---- 4. drift ------------------------------------------------------------------------------------
class DriftAgent(_EvalAgent):
    contract = _contract("eval_drift", "Check whether validation data still looks like the training data.")
    MAX_REFERENCE_ROWS = 20_000

    def execute(self, req: EvalRequest) -> AgentFindings:
        c = self.ctx
        reference = c.reference
        if len(reference) > self.MAX_REFERENCE_ROWS:
            reference = reference.sample(self.MAX_REFERENCE_ROWS, random_state=c.seed)
        table = feature_drift(reference, c.frame, c.numeric, c.categorical)
        result = summarize(table, c.drift_settings)
        evidence = {k: result[k] for k in ("n_features", "n_major", "n_moderate", "share_major", "top_features")}
        names = ", ".join(f"{n} (PSI {v:.2f})" for n, v in result["top_features"][:3])
        if result["verdict"] == "drift":
            finding = Finding(
                agent=self.contract.name, severity="critical", title="Data has drifted",
                detail=f"{result['n_major']} of {result['n_features']} features shifted a lot. Largest: {names}.",
                evidence=evidence,
                hypothesis="Inputs have shifted; prefer features that are stable between train and validation.")
        elif result["n_moderate"] or result["n_major"]:
            finding = Finding(
                agent=self.contract.name, severity="warning", title="Some features are shifting",
                detail=f"{result['n_major']} major and {result['n_moderate']} moderate shifts. Largest: {names}.",
                evidence=evidence)
        else:
            finding = Finding(agent=self.contract.name, severity="info", title="No meaningful drift",
                              detail="Every feature is stable between training and validation.", evidence=evidence)
        return AgentFindings(agent=self.contract.name, summary={**evidence, "verdict": result["verdict"]},
                             findings=[finding])


AGENTS = (PerformanceAgent, SegmentAgent, ErrorAgent, DriftAgent)


# ---- the swarm -----------------------------------------------------------------------------------
@dataclass
class EvaluationReport:
    findings: list[Finding]
    summaries: dict[str, dict]
    failed: dict[str, str]
    title: str = "Evaluation swarm report"
    scope: str = "Validation data only."

    @property
    def verdict(self) -> str:
        severities = {f.severity for f in self.findings}
        return "attention" if "critical" in severities else "watch" if "warning" in severities else "healthy"

    def hypotheses(self, limit: int = 5) -> list[str]:
        """Plain-text ideas for the proposer. They are evidence to read, not commands."""
        seen, out = set(), []
        for finding in self.findings:
            if finding.hypothesis and finding.hypothesis not in seen:
                seen.add(finding.hypothesis)
                out.append(f"{finding.title}: {finding.hypothesis}")
        return out[:limit]

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "failed": self.failed, "summaries": self.summaries,
                "findings": [f.model_dump() for f in self.findings], "hypotheses": self.hypotheses()}

    def markdown(self) -> str:
        lines = [f"# {self.title}", "", f"Overall: **{self.verdict}**. {self.scope}", ""]
        if self.failed:
            lines += ["Agents that failed (their findings are missing):", *[f"- {a}: {e}" for a, e in self.failed.items()], ""]
        for finding in self.findings:
            lines += [f"## [{finding.severity}] {finding.title}", f"_{finding.agent}_", "", finding.detail, ""]
            if finding.hypothesis:
                lines += [f"Idea to test: {finding.hypothesis}", ""]
        return "\n".join(lines)

    def save(self, report_dir: Path, findings_path: Path | None = None) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "evaluation_report.json").write_text(json.dumps(self.to_dict(), indent=2, default=str))
        (report_dir / "evaluation_report.md").write_text(self.markdown())
        if findings_path:
            findings_path.parent.mkdir(parents=True, exist_ok=True)
            findings_path.write_text(json.dumps(self.hypotheses(), indent=2))


class EvaluationSwarm:
    """Runs a set of agents over one shared context. The data swarm reuses this with other agents."""

    agent_classes: tuple = AGENTS
    title = "Evaluation swarm report"
    scope = "Validation data only."
    request_model: type[BaseModel] = EvalRequest

    def __init__(self, context, observer: Observer | None = None, budget: Budget | None = None) -> None:
        self.context = context
        self.observer = observer or Observer()
        self.budget = budget
        self.agents = [cls(context, observer=self.observer, budget=budget) for cls in self.agent_classes]

    def run(self, request: BaseModel | None = None) -> EvaluationReport:
        request = request or self.request_model()
        findings: list[Finding] = []
        summaries: dict[str, dict] = {}
        failed: dict[str, str] = {}
        for agent in self.agents:
            name = agent.contract.name
            try:
                result = agent.run(request)
            except Exception as exc:  # one agent failing must not hide the others
                failed[name] = f"{type(exc).__name__}: {exc}"
                findings.append(Finding(agent=name, severity="warning", title=f"{name} failed",
                                        detail=failed[name]))
                continue
            summaries[name] = result.summary
            findings += result.findings
        for finding in findings:
            metrics.EVAL_FINDINGS.labels(finding.agent, finding.severity).inc()
        findings.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 3))
        return EvaluationReport(findings, summaries, failed, self.title, self.scope)
