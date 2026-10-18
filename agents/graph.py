"""The agent: a LangGraph state machine that runs experiments and learns from them.

    recall -> propose -> novelty -> run -> decide -> record -> (recall ... or END)

  recall   read memory: what was tried on this dataset, recent lessons, current champion
  propose  a proposer (rules now, an LLM later) suggests one ExperimentSpec
  novelty  skip anything memory says was already tried on this data
  run      deterministic training and paired validation comparison (no LLM)
  decide   fixed statistical rule: adopt only a clear improvement (no LLM)
  record   store the outcome in memory and MLflow; adopt = new champion

The proposer is the only place a language model will sit. Everything that touches
data or decides anything is ordinary, tested code.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from functools import wraps
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from agentkit.budget import Budget, BudgetExceeded, KillSwitchTripped
from agentkit.tracing import Observer
from agents.decision import decide
from agents.proposers import Proposer
from agents.spec import ExperimentSpec
from memory.store import MemoryStore
from models import tracking
from observability import agent_metrics
from tools.experiment_runner import ExperimentRunner

MAX_RETRIES = 3
KIND = "agent_experiment"


class AgentState(TypedDict, total=False):
    iteration: int
    max_iterations: int
    retries: int
    champion: dict
    context: dict
    spec: dict | None
    spec_error: str
    result: dict
    verdict: str
    reason: str
    stop_reason: str
    history: Annotated[list, operator.add]


def spec_from_identity(identity: dict, hypothesis: str = "") -> ExperimentSpec:
    """Rebuild a spec from stored data (an identity dict, optionally carrying its hypothesis)."""
    fields = {k: v for k, v in identity.items() if k != "hypothesis"}
    return ExperimentSpec(**fields, hypothesis=hypothesis or identity.get("hypothesis", ""))


def load_champion(store: MemoryStore, runner: ExperimentRunner, dataset_sha256: str) -> ExperimentSpec:
    """The most recently adopted experiment on this data, or the baseline."""
    for entry in store.recent(KIND, limit=1000):
        if entry["dataset_sha256"] == dataset_sha256 and entry["verdict"] == "adopt":
            return spec_from_identity(entry["config"], entry["hypothesis"])
    return runner.baseline_spec()


def build_agent(
    runner: ExperimentRunner,
    store: MemoryStore,
    proposer: Proposer,
    config: dict,
    dataset_sha256: str,
    log: Callable[[str], None] = print,
    budget: Budget | None = None,
    observer: Observer | None = None,
    findings: list[str] | None = None,
):
    budget = budget or Budget.unlimited()
    observer = observer or Observer()
    allowed_groups = list(config["features"]["groups"])
    allowed_windows = list(config["features"]["windows"])
    min_gain = float(config["agent"]["min_gain"])
    track = config.get("tracking", {}).get("enabled", True)

    def tried_keys() -> list[str]:
        keys = []
        for entry in store.recent(KIND, limit=1000):
            if entry["dataset_sha256"] == dataset_sha256:
                keys.append(spec_from_identity(entry["config"]).key())
        return keys

    def tried_summaries() -> list[dict]:
        rows = []
        for entry in store.recent(KIND, limit=40):
            if entry["dataset_sha256"] == dataset_sha256:
                rows.append({"hypothesis": entry["hypothesis"], "verdict": entry["verdict"],
                             "result": entry["conclusion"]})
        return rows

    def recall(state: AgentState) -> dict:
        try:
            budget.check()
        except (BudgetExceeded, KillSwitchTripped) as exc:
            log(f"  stopping: {exc}")
            return {"stop_reason": f"stopped: {exc}"}
        context = {
            "tried_keys": tried_keys(),
            "tried": tried_summaries(),
            "rejections": [],
            "lessons": [lesson["text"] for lesson in store.lessons(limit=8)],
            "findings": list(findings or []),
            "recent": [
                {"hypothesis": e["hypothesis"], "verdict": e["verdict"], "conclusion": e["conclusion"]}
                for e in store.recent(KIND, limit=5)
                if e["dataset_sha256"] == dataset_sha256
            ],
        }
        return {"context": context}

    def with_rejection(state: AgentState, message: str) -> dict:
        """Tell the proposer why its last suggestion was refused, so it can correct itself."""
        context = state["context"]
        return {**context, "rejections": [*context.get("rejections", []), message]}

    def propose(state: AgentState) -> dict:
        champion = spec_from_identity(state["champion"], state["champion"].get("hypothesis", ""))
        retries = state.get("retries", 0) + 1
        try:
            spec = proposer.propose(champion, state["context"])
        except (BudgetExceeded, KillSwitchTripped) as exc:
            log(f"  stopping: {exc}")
            return {"spec": None, "spec_error": "", "stop_reason": f"stopped: {exc}"}
        except Exception as exc:  # noqa: BLE001 - a bad proposal must not crash the loop
            message = f"proposer failed: {exc}"
            log(f"  {message}")
            return {"spec": None, "spec_error": message, "retries": retries,
                    "context": with_rejection(state, message)}
        if spec is None:
            reason = getattr(proposer, "stop_reason", "") or "the proposer has nothing left to suggest"
            return {"spec": None, "spec_error": "", "stop_reason": reason}
        errors = spec.problems(allowed_groups, allowed_windows)
        if errors:
            message = f"{spec.key()} refused: {'; '.join(errors)}"
            log(f"  rejected proposal before running: {'; '.join(errors)}")
            return {"spec": None, "spec_error": message, "retries": retries,
                    "context": with_rejection(state, message)}
        return {"spec": {**spec.identity(), "hypothesis": spec.hypothesis}, "spec_error": ""}

    def route_after_propose(state: AgentState) -> str:
        if state.get("stop_reason"):
            return END
        if state.get("spec") is None:
            return END if state.get("retries", 0) >= MAX_RETRIES else "propose"
        return "novelty"

    def novelty(state: AgentState) -> dict:
        spec = spec_from_identity(state["spec"], state["spec"]["hypothesis"])
        known = store.already_tried(KIND, spec.key(), spec.identity(), dataset_sha256)
        champion_key = spec_from_identity(state["champion"]).key()
        if known or spec.key() == champion_key:
            log(f"  already tried, skipping: {spec.hypothesis}")
            context = with_rejection(state, f"{spec.key()} was already tried; propose something different")
            context["tried_keys"] = [*context["tried_keys"], spec.key()]
            return {"spec": None, "context": context, "retries": state.get("retries", 0) + 1}
        return {}

    def route_after_novelty(state: AgentState) -> str:
        if state.get("spec") is not None:
            return "run"
        return "propose" if state.get("retries", 0) < MAX_RETRIES else END

    def run(state: AgentState) -> dict:
        spec = spec_from_identity(state["spec"], state["spec"]["hypothesis"])
        log(f"  running: {spec.hypothesis}  [{spec.key()}]")
        return {"result": runner.run(spec)}

    def decide_node(state: AgentState) -> dict:
        verdict, reason = decide(state["result"], min_gain)
        return {"verdict": verdict, "reason": reason}

    def record(state: AgentState) -> dict:
        spec = spec_from_identity(state["spec"], state["spec"]["hypothesis"])
        result, verdict, reason = state["result"], state["verdict"], state["reason"]
        metrics = {k: v for k, v in result.items() if isinstance(v, int | float)}
        metrics["diff_ci_low"], metrics["diff_ci_high"] = result["diff_ci"]
        exp_id, _ = store.record_experiment(
            KIND,
            spec.key(),
            spec.identity(),
            metrics,
            hypothesis=spec.hypothesis,
            verdict=verdict,
            conclusion=reason,
            dataset_sha256=dataset_sha256,
        )
        budget.charge_experiment()
        agent_metrics.EXPERIMENTS.labels(verdict).inc()
        if verdict == "adopt":
            agent_metrics.BEST_PR_AUC.set(result["val_pr_auc"])
            store.add_lesson(f"Adopted: {spec.hypothesis}. {reason}", experiment_id=exp_id, tags="agent adopt")
        elif result["diff_ci"][1] < 0:
            store.add_lesson(f"Does not help: {spec.hypothesis}. {reason}", experiment_id=exp_id, tags="agent worse")
        if track:
            tracking.log_agent_candidate(config, spec.identity(), spec.hypothesis, result, verdict, reason)

        update: dict = {
            "iteration": state["iteration"] + 1,
            "retries": 0,
            "history": [{"hypothesis": spec.hypothesis, "key": spec.key(), "verdict": verdict, "reason": reason}],
        }
        if verdict == "adopt":
            runner.set_champion(spec)
            update["champion"] = {**spec.identity(), "hypothesis": spec.hypothesis}
        log(f"  -> {verdict.upper()}: {reason}")
        return update

    def route_after_record(state: AgentState) -> str:
        return END if state["iteration"] >= state["max_iterations"] else "recall"

    def route_after_recall(state: AgentState) -> str:
        return END if state.get("stop_reason") else "propose"

    def observed(name: str, fn):
        @wraps(fn)
        def wrapper(state: AgentState) -> dict:
            with observer.step(f"agent.{name}", iteration=state.get("iteration", 0)) as outcome:
                update = fn(state)
                if name == "decide":
                    outcome["verdict"] = update.get("verdict")
                if name == "propose" and update.get("spec"):
                    outcome["hypothesis"] = update["spec"]["hypothesis"][:120]
                if update.get("stop_reason"):
                    outcome["stop_reason"] = update["stop_reason"][:160]
                return update

        return wrapper

    graph = StateGraph(AgentState)
    graph.add_node("recall", observed("recall", recall))
    graph.add_node("propose", observed("propose", propose))
    graph.add_node("novelty", observed("novelty", novelty))
    graph.add_node("run", observed("run", run))
    graph.add_node("decide", observed("decide", decide_node))
    graph.add_node("record", observed("record", record))
    graph.set_entry_point("recall")
    graph.add_conditional_edges("recall", route_after_recall, {"propose": "propose", END: END})
    graph.add_conditional_edges("propose", route_after_propose, {"propose": "propose", "novelty": "novelty", END: END})
    graph.add_conditional_edges("novelty", route_after_novelty, {"propose": "propose", "run": "run", END: END})
    graph.add_edge("run", "decide")
    graph.add_edge("decide", "record")
    graph.add_conditional_edges("record", route_after_record, {"recall": "recall", END: END})
    return graph.compile()
