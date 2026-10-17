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
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from agents.decision import decide
from agents.proposers import Proposer
from agents.spec import ExperimentSpec
from memory.store import MemoryStore
from models import tracking
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
):
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

    def recall(state: AgentState) -> dict:
        context = {
            "tried_keys": tried_keys(),
            "lessons": [lesson["text"] for lesson in store.lessons(limit=8)],
            "recent": [
                {"hypothesis": e["hypothesis"], "verdict": e["verdict"], "conclusion": e["conclusion"]}
                for e in store.recent(KIND, limit=5)
                if e["dataset_sha256"] == dataset_sha256
            ],
        }
        return {"context": context}

    def propose(state: AgentState) -> dict:
        champion = spec_from_identity(state["champion"], state["champion"].get("hypothesis", ""))
        try:
            spec = proposer.propose(champion, state["context"])
        except Exception as exc:  # noqa: BLE001 - a bad proposal must not crash the loop
            return {"spec": None, "spec_error": f"proposer failed: {exc}", "retries": state.get("retries", 0) + 1}
        if spec is None:
            return {"spec": None, "spec_error": "", "stop_reason": "the proposer has nothing left to suggest"}
        errors = spec.problems(allowed_groups, allowed_windows)
        if errors:
            log(f"  rejected proposal before running: {'; '.join(errors)}")
            return {"spec": None, "spec_error": "; ".join(errors), "retries": state.get("retries", 0) + 1}
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
            context = {**state["context"], "tried_keys": [*state["context"]["tried_keys"], spec.key()]}
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
        if verdict == "adopt":
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

    graph = StateGraph(AgentState)
    graph.add_node("recall", recall)
    graph.add_node("propose", propose)
    graph.add_node("novelty", novelty)
    graph.add_node("run", run)
    graph.add_node("decide", decide_node)
    graph.add_node("record", record)
    graph.set_entry_point("recall")
    graph.add_edge("recall", "propose")
    graph.add_conditional_edges("propose", route_after_propose, {"propose": "propose", "novelty": "novelty", END: END})
    graph.add_conditional_edges("novelty", route_after_novelty, {"propose": "propose", "run": "run", END: END})
    graph.add_edge("run", "decide")
    graph.add_edge("decide", "record")
    graph.add_conditional_edges("record", route_after_record, {"recall": "recall", END: END})
    return graph.compile()