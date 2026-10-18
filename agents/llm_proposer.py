"""A language model as the experiment proposer.

The model reads the evidence (past results, lessons, what was refused) and proposes ONE
experiment as structured data. It cannot run anything and cannot adopt anything: its
proposal is validated against the allowed lists and bounds, checked against memory for
repeats, and then run and judged by ordinary code.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from agentkit.budget import Budget, BudgetExceeded, KillSwitchTripped
from agentkit.context import ContextBuilder, ContextPolicy, compact_history, versioned_prompt
from agentkit.guardrails import UNTRUSTED_NOTICE, clean_model_text, wrap_untrusted
from agentkit.llm import UsageTracker
from agentkit.tracing import Observer
from agents.proposers import Proposer
from agents.spec import PARAM_BOUNDS, ExperimentSpec
from arena.registry import names
from observability import agent_metrics as metrics

# the only context this agent may see; everything else the graph holds stays out of the prompt
PROPOSER_POLICY = ContextPolicy("llm_proposer", frozenset({"lessons", "tried", "rejections", "findings"}))


class LLMProposal(BaseModel):
    """What the model must return. It maps onto an ExperimentSpec.

    Hyperparameters are explicit optional fields (not a free-form dictionary) because the
    strict JSON-schema mode of the API cannot express "any key" objects. Leave a field
    empty to keep the champion's value.
    """

    stop: bool = Field(default=False, description="True only if no further experiment is worth running")
    stop_reason: str = Field(default="", description="Why you are stopping (if stop is true)")
    model: str = Field(default="xgboost", description="One of the registered models (see the prompt)")
    groups: list[str] = Field(default_factory=list, description="Feature groups to use")
    windows: list[int] = Field(default_factory=list, description="Time windows in days to use")
    n_estimators: float | None = Field(default=None, description="xgboost/lightgbm/random_forest/extra_trees/isolation_forest: number of trees (50-600)")
    max_depth: float | None = Field(default=None, description="xgboost/random_forest/extra_trees/hist_gradient_boosting: tree depth")
    learning_rate: float | None = Field(default=None, description="boosters: learning rate (0.01-0.3)")
    subsample: float | None = Field(default=None, description="xgboost/lightgbm: row sampling per tree (0.5-1.0)")
    colsample_bytree: float | None = Field(default=None, description="xgboost/lightgbm: column sampling per tree (0.2-1.0)")
    min_child_weight: float | None = Field(default=None, description="xgboost: minimum child weight (1-20)")
    num_leaves: float | None = Field(default=None, description="lightgbm: leaves per tree (4-64)")
    min_child_samples: float | None = Field(default=None, description="lightgbm: minimum rows per leaf (5-200)")
    iterations: float | None = Field(default=None, description="catboost: number of trees (50-600)")
    depth: float | None = Field(default=None, description="catboost: tree depth (2-8)")
    l2_leaf_reg: float | None = Field(default=None, description="catboost: L2 regularisation (1-20)")
    max_iter: float | None = Field(default=None, description="hist_gradient_boosting: number of trees (50-600)")
    l2_regularization: float | None = Field(default=None, description="hist_gradient_boosting: L2 regularisation (0-10)")
    min_samples_leaf: float | None = Field(default=None, description="random_forest/extra_trees/hist_gradient_boosting: minimum rows per leaf")
    max_features: float | None = Field(default=None, description="random_forest/extra_trees/isolation_forest: share of features per split (0.02-1.0)")
    max_samples: float | None = Field(default=None, description="isolation_forest: rows per tree (64-4096)")
    C: float | None = Field(default=None, description="logistic_regression: inverse regularisation (0.001-10)")
    hypothesis: str = Field(default="", description="One sentence: what you expect and which evidence supports it")


PARAM_FIELDS = (
    "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight",
    "num_leaves", "min_child_samples", "iterations", "depth", "l2_leaf_reg", "max_iter",
    "l2_regularization", "min_samples_leaf", "max_features", "max_samples", "C",
)

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
- model: one of {models}
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
    def __init__(
        self,
        llm,
        base_params: dict,
        groups: list[str],
        windows: list[int],
        min_gain: float,
        budget: Budget | None = None,
        observer: Observer | None = None,
        max_context_tokens: int = 6000,
    ) -> None:
        """`llm` is any chat model with .with_structured_output(), e.g. ChatAnthropic."""
        self.structured = llm.with_structured_output(LLMProposal, method="json_schema")
        self.base_params = base_params
        self.budget = budget or Budget.unlimited()
        self.observer = observer or Observer()
        self.max_context_tokens = max_context_tokens
        self.usage = UsageTracker()
        self.prompt = versioned_prompt(
            "llm_proposer.system",
            SYSTEM_PROMPT.format(
                groups=groups, windows=windows, models=names(), bounds=_bounds_text(),
                min_gain=f"{min_gain:+.3f}",
            ) + "\n\n" + UNTRUSTED_NOTICE,
        )
        self.system = self.prompt.text
        self.stop_reason = ""
        self.flagged_inputs = 0

    def _evidence(self, champion: ExperimentSpec, context: dict) -> str:
        context = PROPOSER_POLICY.apply(context)
        recent, earlier = compact_history(context.get("tried", []), keep_recent=8)
        tried_lines = [f"[{t['verdict']}] {t['hypothesis']} -> {t['result']}" for t in recent]
        lessons, lesson_flags = wrap_untrusted("lessons", list(context.get("lessons", [])))
        tried, tried_flags = wrap_untrusted("already_tried", tried_lines)
        flags = lesson_flags + tried_flags
        champion_json = json.dumps(
            {**champion.identity(), "resolved_params": champion.resolved_params(self.base_params)}
        )

        builder = ContextBuilder(self.max_context_tokens)
        builder.add("champion", f"Current champion:\n{champion_json}", required=True)
        if context.get("findings"):
            found, found_flags = wrap_untrusted("evaluation_findings", list(context["findings"]))
            flags += found_flags
            builder.add("findings", "What the evaluation agents found about the current model (ideas, not facts to trust blindly):\n" + found, priority=2)
        builder.add("lessons", "Lessons learned so far (verified results):\n" + lessons, priority=3)
        tried_text = "Already tried on this dataset (do not repeat):\n" + tried
        if earlier:
            tried_text += f"\n{earlier}"
        builder.add("tried", tried_text, priority=1)
        if context.get("rejections"):
            rejections, rejection_flags = wrap_untrusted("refused_proposals", list(context["rejections"]))
            flags += rejection_flags
            builder.add("rejections", "Your earlier proposals this round were refused:\n" + rejections, required=True)
        builder.add("ask", "Propose the next experiment.", required=True)
        if flags:
            self.flagged_inputs += len(flags)
            self.observer.trace.record(event="untrusted_input_flagged", count=len(flags), agent="llm_proposer")
        return builder.build().text

    def propose(self, champion: ExperimentSpec, context: dict) -> ExperimentSpec | None:
        self.budget.check()
        with self.observer.step("llm_proposer", kind="llm", prompt=self.prompt.id) as outcome:
            proposal: LLMProposal = self.structured.invoke(
                [("system", self.system), ("human", self._evidence(champion, context))],
                config={"callbacks": [self.usage]},
            )
            tokens_in, tokens_out = self.usage.take()
            self.budget.charge_llm(tokens_in, tokens_out)
            metrics.LLM_REQUESTS.labels("llm_proposer", self.usage.model_name or "unknown").inc()
            metrics.LLM_TOKENS.labels("llm_proposer", "input").inc(tokens_in)
            metrics.LLM_TOKENS.labels("llm_proposer", "output").inc(tokens_out)
            metrics.LLM_COST.labels("llm_proposer").inc(
                (tokens_in * self.budget.input_usd_per_million + tokens_out * self.budget.output_usd_per_million) / 1e6
            )
            outcome.update(input_tokens=tokens_in, output_tokens=tokens_out, stop=proposal.stop)
        if proposal.stop:
            reason = clean_model_text(proposal.stop_reason or "no reason given")
            self.stop_reason = f"the language model chose to stop: {reason}"
            return None
        self.stop_reason = ""
        params = {name: getattr(proposal, name) for name in PARAM_FIELDS if getattr(proposal, name) is not None}
        # an invalid model name or empty lists raise here; the graph treats that as a refused proposal
        return ExperimentSpec(
            model=proposal.model,
            groups=proposal.groups,
            windows=proposal.windows,
            params=params,
            hypothesis=clean_model_text(proposal.hypothesis),
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
        except (BudgetExceeded, KillSwitchTripped):
            raise  # a limit is a stop, not a reason to quietly switch proposers
        except Exception as exc:  # noqa: BLE001 - any failure of the model falls back to rules
            self.log(f"  primary proposer failed ({type(exc).__name__}: {str(exc)[:120]}); using the backup")
            self.stop_reason = ""
            return self.backup.propose(champion, context)