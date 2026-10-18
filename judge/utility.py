"""Pick the model to serve by what it is worth, and say why.

Ranking by PR-AUC alone left six models "tied", and the old rule (highest number wins) then
picked one by noise. The judge replaces that with an explicit, written policy:

  1. Hard constraints. A model that failed is out. If explanations are required (the API returns
     reasons for each alert), a model that cannot explain its scores is out.
  2. Utility. At the operating point (the top `review_fraction` of accounts go to a reviewer) each
     model earns   caught_fraud * fraud_value  -  reviewed * review_cost  -  false_alarms * false_alarm_cost,
     reported per 10,000 accounts scored. The prices are assumptions in configs/models.yaml.
  3. Statistics. A paired bootstrap on the same resampled accounts gives each model's utility gap to
     the best one. A model is "tied" when its interval for that gap includes zero.
  4. Preference among ties. Tied models are equal within noise, so pick the cheapest to operate:
     fastest to retrain, then smallest. A tied model whose point utility is more than
     `max_point_loss` below the best is not considered, so noise cannot hide a real shortfall.

The judge is deterministic code. Nothing in it reads the sealed test split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from agentkit.harness import AgentContract, BaseAgent

EXPLAINABLE = frozenset({"xgboost", "lightgbm", "catboost"})  # models whose scores the API can explain
PER = 10_000


@dataclass(frozen=True)
class UtilityConfig:
    fraud_value: float = 400.0  # money saved for each fraud caught
    review_cost: float = 6.0  # analyst time for every reviewed account
    false_alarm_cost: float = 3.0  # extra customer friction for each wrongly flagged account
    require_explanations: bool = True
    max_point_loss: float = 0.10  # a tied model may be at most this share below the best utility

    @classmethod
    def from_config(cls, config: dict) -> UtilityConfig:
        known = {k: v for k, v in (config.get("judge") or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Candidate:
    name: str
    val_probs: np.ndarray
    train_seconds: float = 0.0
    size_mb: float = 0.0
    explainable: bool | None = None  # None: look it up in EXPLAINABLE

    @property
    def can_explain(self) -> bool:
        return self.name in EXPLAINABLE if self.explainable is None else self.explainable


def utility_per_10k(caught: float, reviewed: int, n: int, cfg: UtilityConfig) -> float:
    false_alarms = reviewed - caught
    total = caught * cfg.fraud_value - reviewed * cfg.review_cost - false_alarms * cfg.false_alarm_cost
    return float(total / n * PER)


def review_depth(n: int, review_fraction: float) -> int:
    return max(1, int(round(n * review_fraction)))


def _caught_in_top(y: np.ndarray, p: np.ndarray, k: int) -> float:
    top = np.argpartition(-p, k - 1)[:k]
    return float(y[top].sum())


@dataclass
class Verdict:
    winner: str
    rows: list[dict]
    rationale: str
    config: UtilityConfig = field(repr=False, default_factory=UtilityConfig)

    def to_dict(self) -> dict:
        return {"winner": self.winner, "rationale": self.rationale, "rows": self.rows,
                "prices": self.config.__dict__}

    def markdown(self) -> str:
        c = self.config
        lines = [
            "# Judge verdict (validation snapshots only)", "",
            f"Prices assumed: {c.fraud_value:g} per fraud caught, {c.review_cost:g} per reviewed account, "
            f"{c.false_alarm_cost:g} extra per false alarm. Utility is per {PER:,} accounts scored.", "",
            f"**Served model: {self.winner}.** {self.rationale}", "",
            "| Model | Utility [95% interval] | Gap to best [95% interval] | Train s | Size MB | Explains | Verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in self.rows:
            if r["status"] == "excluded":
                lines.append(f"| {r['model']} | - | - | {r['train_seconds']:.1f} | {r['size_mb']:.1f} | "
                             f"{'yes' if r['explains'] else 'no'} | excluded: {r['reason']} |")
                continue
            lines.append(
                f"| {r['model']} | {r['utility']:,.0f} [{r['utility_ci95'][0]:,.0f}, {r['utility_ci95'][1]:,.0f}] | "
                f"{r['gap_vs_best']:+,.0f} [{r['gap_ci95'][0]:+,.0f}, {r['gap_ci95'][1]:+,.0f}] | "
                f"{r['train_seconds']:.1f} | {r['size_mb']:.1f} | {'yes' if r['explains'] else 'no'} | {r['status']} |")
        return "\n".join(lines) + "\n"


def judge(
    candidates: list[Candidate], y: np.ndarray, cfg: UtilityConfig, review_fraction: float,
    n_boot: int = 200, seed: int = 42,
) -> Verdict:
    n = len(y)
    k = review_depth(n, review_fraction)
    rows: dict[str, dict] = {}
    eligible: list[Candidate] = []
    for cand in candidates:
        base = {"model": cand.name, "train_seconds": cand.train_seconds, "size_mb": cand.size_mb,
                "explains": cand.can_explain}
        if cfg.require_explanations and not cand.can_explain:
            rows[cand.name] = {**base, "status": "excluded", "reason": "cannot explain its scores"}
        else:
            rows[cand.name] = base
            eligible.append(cand)
    relaxed = ""
    if not eligible:  # nothing satisfies the constraint: judge everyone and say so
        relaxed = " No model could explain its scores, so that requirement was waived."
        eligible = list(candidates)
        for cand in eligible:
            rows[cand.name] = {"model": cand.name, "train_seconds": cand.train_seconds,
                               "size_mb": cand.size_mb, "explains": cand.can_explain}
    if not eligible:
        raise ValueError("no candidates to judge")

    point = {c.name: utility_per_10k(_caught_in_top(y, c.val_probs, k), k, n, cfg) for c in eligible}
    best = max(point, key=point.get)
    rng = np.random.default_rng(seed)
    draws = {c.name: np.empty(n_boot) for c in eligible}
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        for c in eligible:
            draws[c.name][i] = utility_per_10k(_caught_in_top(yb, c.val_probs[idx], k), k, n, cfg)

    tied = []
    for c in eligible:
        gap = draws[c.name] - draws[best]
        low, high = np.percentile(gap, [2.5, 97.5])
        ci_u = np.percentile(draws[c.name], [2.5, 97.5])
        within_noise = c.name == best or high >= 0
        within_loss = point[c.name] >= point[best] * (1 - cfg.max_point_loss)
        if c.name == best:
            status = "best utility"
        elif not within_noise:
            status = "worse"
        elif not within_loss:
            status = f"tied but more than {cfg.max_point_loss:.0%} below best"
        else:
            status = "tied with best"
            tied.append(c)
        rows[c.name].update({"utility": point[c.name], "utility_ci95": [float(ci_u[0]), float(ci_u[1])],
                             "gap_vs_best": point[c.name] - point[best], "gap_ci95": [float(low), float(high)],
                             "status": status})
    pool = [next(c for c in eligible if c.name == best), *tied]
    winner = min(pool, key=lambda c: (round(c.train_seconds, 1), c.size_mb, -point[c.name]))
    rows[winner.name]["status"] += " - SERVED"

    if len(pool) == 1:
        rationale = f"{winner.name} has the highest utility and no other model is statistically tied with it.{relaxed}"
    else:
        others = ", ".join(c.name for c in pool if c.name != winner.name)
        rationale = (f"{len(pool)} models are tied on utility within noise ({winner.name}, {others}). "
                     f"{winner.name} is the cheapest to operate: {winner.train_seconds:.0f}s to train, "
                     f"{winner.size_mb:.1f} MB.{relaxed}")
    ordered = sorted(rows.values(), key=lambda r: (r["status"] == "excluded", -r.get("utility", 0.0)))
    return Verdict(winner.name, ordered, rationale, cfg)


# ---- the judge as a traced agent ---------------------------------------------------------------
class JudgeRequest(BaseModel):
    n_boot: int = Field(default=200, ge=20, le=5000)


class JudgeOutput(BaseModel):
    winner: str
    rationale: str
    rows: list[dict]


class JudgeAgent(BaseAgent):
    contract = AgentContract(
        name="judge", responsibility="Decide which candidate model to serve, by utility, with stated reasons.",
        kind="deterministic", input_model=JudgeRequest, output_model=JudgeOutput, timeout_seconds=600.0)

    def __init__(self, candidates: list[Candidate], y: np.ndarray, cfg: UtilityConfig,
                 review_fraction: float, seed: int = 42, **kwargs) -> None:
        super().__init__(**kwargs)
        self.candidates, self.y, self.cfg = candidates, y, cfg
        self.review_fraction, self.seed = review_fraction, seed
        self.verdict: Verdict | None = None

    def execute(self, req: JudgeRequest) -> JudgeOutput:
        self.verdict = judge(self.candidates, self.y, self.cfg, self.review_fraction, req.n_boot, self.seed)
        return JudgeOutput(winner=self.verdict.winner, rationale=self.verdict.rationale, rows=self.verdict.rows)
