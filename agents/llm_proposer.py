"""A language model as the experiment proposer.

The model reads the evidence (past results, lessons, what was refused) and proposes ONE
experiment as structured data. It cannot run anything and cannot adopt anything: its
proposal is validated against the allowed lists and bounds, checked against memory for
repeats, and then run and judged by ordinary code.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from agents.proposers import Proposer
from agents.spec import PARAM_BOUNDS, ExperimentSpec


class LLMProposal(BaseModel):
    """What the model must return. It maps onto an ExperimentSpec.

    Hyperparameters are explicit optional fields (not a free-form dictionary) because the
    strict JSON-schema mode of the API cannot express "any key" objects. Leave a field
    empty to keep the champion's value.
    """

    stop: bool = Field(default=False, description="True only if no further experiment is worth running")
    stop_reason: str = Field(default="", description="Why you are stopping (if stop is true)")
    model: str = Field(default="xgboost", description="xgboost or logistic_regression")
    groups: list[str] = Field(default_factory=list, description="Feature groups to use")
    windows: list[int] = Field(default_factory=list, description="Time windows in days to use")
    n_estimators: float | None = Field(default=None, description="xgboost: number of trees (50-600)")
    max_depth: float | None = Field(default=None, description="xgboost: tree depth (2-8)")
    learning_rate: float | None = Field(default=None, description="xgboost: learning rate (0.01-0.3)")
    subsample: float | None = Field(default=None, description="xgboost: row sampling per tree (0.5-1.0)")
    colsample_bytree: float | None = Field(default=None, description="xgboost: column sampling per tree (0.2-1.0)")
    min_child_weight: float | None = Field(default=None, description="xgboost: minimum child weight (1-20)")
    C: float | None = Field(default=None, description="logistic_regression: inverse regularisation (0.001-10)")
    hypothesis: str = Field(default="", description="One sentence: what you expect and which evidence supports it")


PARAM_FIELDS = ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
                "min_child_weight", "C")

SYSTEM_PROMPT = """You are the research planner for a telecom fraud-risk model.

The model scores subscribers monthly for the chance that fraud is confirmed within 30 days.
Fraud is rare (about 1% of rows), so the metric is validation PR-AUC. You propose exactly ONE
experiment at a time. Your proposal is run by separate code and judged by a fixed statistical
rule: it is adopted only if validation PR-AUC improves clearly (the confidence interval of the
paired difference must be above zero and at least {min_gain}). You cannot change that rule,
see the test data, or run anything yourself.

You may only choose from:
- feature groups: {groups}
- windows (days): {windows}
- model: xgboost or logistic_regression
- hyperparameter fields (leave empty to keep the champion's value), each within these bounds: {bounds}

How to propose well:
- Base the idea on the evidence you are given and say which evidence in `hypothesis`.
- Never repeat an experiment listed under "already tried", and do not repeat refused proposals.
- Prefer one clear change over many at once, so the result is interpretable.
- If the evidence says further experiments are unlikely to help, set stop=true and explain why.
Return the proposal in the required structure and nothing else."""


def _bounds_text() -> str:
    return json.dumps({m: {k: [v[0], v[1]] for k, v in p.items()} for m, p in PARAM_BOUNDS.items()})


class LLMProposer:
    def __init__(self, llm, base_params: dict, groups: list[str], windows: list[int], min_gain: float) -> None:
        """`llm` is any chat model with .with_structured_output(), e.g. ChatAnthropic."""
        self.structured = llm.with_structured_output(LLMProposal, method="json_schema")
        self.base_params = base_params
        self.system = SYSTEM_PROMPT.format(
            groups=groups, windows=windows, bounds=_bounds_text(), min_gain=f"{min_gain:+.3f}"
        )
        self.stop_reason = ""

    def _evidence(self, champion: ExperimentSpec, context: dict) -> str:
        lessons = [f"- {text}" for text in context.get("lessons", [])] or ["- none yet"]
        tried = [
            f"- [{t['verdict']}] {t['hypothesis']} -> {t['result']}" for t in context.get("tried", [])
        ] or ["- nothing yet"]
        champion_json = json.dumps(
            {**champion.identity(), "resolved_params": champion.resolved_params(self.base_params)}
        )
        parts = [
            "Current champion:",
            champion_json,
            "",
            "Lessons learned so far (verified results):",
            *lessons,
            "",
            "Already tried on this dataset (do not repeat):",
            *tried,
        ]
        if context.get("rejections"):
            parts += ["", "Your earlier proposals this round were refused:"]
            parts += [f"- {reason}" for reason in context["rejections"]]
        parts += ["", "Propose the next experiment."]
        return "\n".join(parts)

    def propose(self, champion: ExperimentSpec, context: dict) -> ExperimentSpec | None:
        proposal: LLMProposal = self.structured.invoke(
            [("system", self.system), ("human", self._evidence(champion, context))]
        )
        if proposal.stop:
            self.stop_reason = f"the language model chose to stop: {proposal.stop_reason or 'no reason given'}"
            return None
        self.stop_reason = ""
        params = {name: getattr(proposal, name) for name in PARAM_FIELDS if getattr(proposal, name) is not None}
        # an invalid model name or empty lists raise here; the graph treats that as a refused proposal
        return ExperimentSpec(
            model=proposal.model,
            groups=proposal.groups,
            windows=proposal.windows,
            params=params,
            hypothesis=proposal.hypothesis,
        )


class FallbackProposer:
    """Use the primary proposer; if it errors (network, API, bad output), use the backup."""

    def __init__(self, primary: Proposer, backup: Proposer, log=print) -> None:
        self.primary = primary
        self.backup = backup
        self.log = log
        self.stop_reason = ""

    def propose(self, champion: ExperimentSpec, context: dict) -> ExperimentSpec | None:
        try:
            spec = self.primary.propose(champion, context)
            self.stop_reason = getattr(self.primary, "stop_reason", "")
            return spec
        except Exception as exc:  # noqa: BLE001 - any failure of the model falls back to rules
            self.log(f"  primary proposer failed ({type(exc).__name__}: {str(exc)[:120]}); using the backup")
            self.stop_reason = ""
            return self.backup.propose(champion, context)