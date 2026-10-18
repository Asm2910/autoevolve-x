"""The mission graph: plan, data review, evaluate, arena + judge, improve, reflect.

One LangGraph state machine. A `dispatch` node reads the plan's list of pending steps and routes to
the step to run next; every step returns to `dispatch`. That keeps the order in the plan (data) rather
than in the wiring, so a re-plan or a second improvement round is only a change to the list.

The graph does no modelling itself. It calls `Services`, small functions that wrap the swarms, the
arena, the judge and the improvement agent. Tests pass in fakes; `mission/services.py` has the real ones.
A step that raises is recorded and the mission continues, so one broken part never hides the rest.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from agentkit.budget import Budget, BudgetExceeded, KillSwitchTripped
from agentkit.tracing import Observer
from mission.plan import Mission, MissionPlanner, Plan
from mission.reflection import reflect


@dataclass
class Services:
    data_review: Callable[[], dict]
    evaluate: Callable[[], dict]
    arena: Callable[[], dict]
    improve: Callable[[int, list[str]], dict]
    remember: Callable[[str, str], None] = lambda text, tags: None


class MissionState(TypedDict, total=False):
    mission: dict
    plan: dict
    pending: list[str]
    results: dict
    history: Annotated[list, operator.add]
    timeline: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]
    round: int
    last_round_adopted: int
    halted: str
    reflection: dict


def build_mission_graph(services: Services, planner: MissionPlanner | None = None, budget: Budget | None = None,
                        observer: Observer | None = None, log: Callable[[str], None] = print):
    planner = planner or MissionPlanner()
    budget = budget or Budget.unlimited()
    observer = observer or Observer()

    def mission_of(state: MissionState) -> Mission:
        return Mission(**state["mission"])

    def timed(name: str, fn: Callable[[MissionState], dict]):
        def node(state: MissionState) -> dict:
            started = time.perf_counter()
            update: dict = {}
            status = "ok"
            try:
                with observer.step(f"mission.{name}", round=state.get("round", 0)):
                    update = fn(state)
            except (BudgetExceeded, KillSwitchTripped) as exc:
                log(f"  mission stopped: {exc}")
                return {"halted": f"stopped: {exc}", "pending": [] if name == "reflect" else ["reflect"],
                        "timeline": [{"step": name, "status": "stopped", "seconds": round(time.perf_counter() - started, 1)}]}
            except Exception as exc:  # noqa: BLE001 - one broken step must not end the mission
                status = f"failed: {type(exc).__name__}: {str(exc)[:160]}"
                log(f"  step {name} {status}")
                update = {"errors": [f"{name}: {status}"], "pending": list(state.get("pending", []))[1:]}
            entry = {"step": name, "status": status, "seconds": round(time.perf_counter() - started, 1)}
            return {**update, "timeline": [entry, *update.get("timeline", [])]}
        return node

    def start(state: MissionState) -> dict:
        mission = mission_of(state)
        plan = planner.plan(mission)
        log(f"Plan: {' -> '.join(plan.names())}. {plan.rationale}")
        return {"plan": plan.model_dump(), "pending": plan.names(), "round": 0, "results": {}, "last_round_adopted": 0}

    def dispatch(state: MissionState) -> dict:
        return {}

    def route(state: MissionState) -> str:
        pending = state.get("pending") or []
        return pending[0] if pending else END

    def pop(state: MissionState) -> list[str]:
        return list(state["pending"][1:])

    def data_review(state: MissionState) -> dict:
        budget.check()
        log("Step: data review")
        report = services.data_review()
        results = {**state["results"], "data_review": report}
        update: dict = {"results": results, "pending": pop(state)}
        if report.get("verdict") == "attention":
            new = planner.replan(Plan(**state["plan"]), mission_of(state), report, None)
            log(f"  critical data problems. {new.rationale}")
            update |= {"plan": new.model_dump(), "pending": [s for s in new.names() if s != "data_review"], "halted": new.rationale}
        return update

    def evaluate(state: MissionState) -> dict:
        budget.check()
        log("Step: evaluate the current model")
        report = services.evaluate()
        results = {**state["results"], "evaluate": report}
        new = planner.replan(Plan(**state["plan"]), mission_of(state), state["results"].get("data_review"), report)
        log(f"  re-planned: {new.rationale}")
        remaining = [n for n in new.names() if n in state["pending"][1:]]
        return {"results": results, "plan": new.model_dump(), "pending": remaining}

    def arena(state: MissionState) -> dict:
        budget.check()
        log("Step: model arena and judge")
        return {"results": {**state["results"], "arena": services.arena()}, "pending": pop(state)}

    def improve(state: MissionState) -> dict:
        budget.check()
        plan = Plan(**state["plan"])
        params = (plan.step("improve") or plan.steps[0]).params
        round_number = state.get("round", 0) + 1
        log(f"Step: improvement round {round_number} ({params.get('iterations')} experiments)")
        outcome = services.improve(int(params.get("iterations", 2)), list(params.get("focus", [])))
        history = outcome.get("history", [])
        adopted = sum(1 for h in history if h["verdict"] == "adopt")
        first = state["results"].get("improve", {}).get("start_pr_auc")  # keep round 1's start, not the latest round's
        if first is not None:
            outcome = {**outcome, "start_pr_auc": first}
        results = {**state["results"], "improve": outcome}
        return {"results": results, "history": history, "round": round_number, "last_round_adopted": adopted,
                "pending": pop(state)}

    def reflect_node(state: MissionState) -> dict:
        mission = mission_of(state)
        log("Step: reflection")
        reflection = reflect(state["results"], state.get("history", []), state.get("round", 0), mission.rounds,
                             state.get("last_round_adopted", 0))
        for text, tags in reflection.lessons:
            services.remember(text, tags)
        update = {"reflection": {"lessons": [t for t, _ in reflection.lessons], "next_steps": reflection.next_steps,
                                 "summary": reflection.summary, "again": reflection.again,
                                 "adopted": reflection.adopted}, "pending": pop(state)}
        if reflection.again and not state.get("halted"):
            log("  reflection: the last round adopted a change, so another round will run")
            update["pending"] = ["improve", "reflect"]
        return update

    graph = StateGraph(MissionState)
    graph.add_node("start", timed("plan", start))
    graph.add_node("dispatch", dispatch)
    for name, fn in (("data_review", data_review), ("evaluate", evaluate), ("arena", arena),
                     ("improve", improve), ("reflect", reflect_node)):
        graph.add_node(name, timed(name, fn))
    graph.set_entry_point("start")
    graph.add_edge("start", "dispatch")
    targets = {n: n for n in ("data_review", "evaluate", "arena", "improve", "reflect")} | {END: END}
    graph.add_conditional_edges("dispatch", route, targets)
    for name in ("data_review", "evaluate", "arena", "improve", "reflect"):
        graph.add_edge(name, "dispatch")
    return graph.compile()


def run_mission(mission: Mission, services: Services, **kwargs) -> MissionState:
    graph = build_mission_graph(services, **kwargs)
    return graph.invoke({"mission": mission.model_dump(), "history": [], "timeline": [], "errors": []},
                        config={"recursion_limit": 100})
