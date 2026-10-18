"""The mission graph, planner and reflection, driven by fake services (no modelling needed)."""

import pytest

from agentkit.budget import Budget, BudgetExceeded
from mission.graph import Services, run_mission
from mission.plan import Mission, MissionPlanner, Plan
from mission.reflection import reflect
from mission.report import mission_markdown, mission_payload


def finding(severity, title):
    return {"agent": "x", "severity": severity, "title": title, "detail": "", "evidence": {}, "hypothesis": ""}


class Fake:
    def __init__(self, data_verdict="watch", eval_verdict="watch", adopt_rounds=(1,), fail=None, ideas=("Try deeper trees",)):
        self.calls, self.lessons, self.round = [], [], 0
        self.data_verdict, self.eval_verdict, self.adopt_rounds, self.fail, self.ideas = data_verdict, eval_verdict, set(adopt_rounds), fail, list(ideas)

    def maybe_fail(self, name):
        if self.fail == name:
            raise RuntimeError(f"{name} exploded")

    def data_review(self):
        self.calls.append("data_review")
        self.maybe_fail("data_review")
        findings = [finding("critical", "Duplicate labels")] if self.data_verdict == "attention" else [finding("info", "ok")]
        return {"verdict": self.data_verdict, "findings": findings, "hypotheses": []}

    def evaluate(self):
        self.calls.append("evaluate")
        self.maybe_fail("evaluate")
        return {"verdict": self.eval_verdict, "findings": [finding("warning", "Weaker on region=east")], "hypotheses": self.ideas, "pr_auc": 0.3}

    def arena(self):
        self.calls.append("arena")
        self.maybe_fail("arena")
        return {"winner": "xgboost", "rationale": "3 models tied; xgboost is cheapest", "rows": []}

    def improve(self, iterations, focus):
        self.round += 1
        self.calls.append(f"improve:{iterations}:{len(focus)}")
        self.maybe_fail("improve")
        verdict = "adopt" if self.round in self.adopt_rounds else "reject"
        return {"history": [{"hypothesis": f"idea {self.round}", "verdict": verdict, "reason": "PR-AUC change +0.02"}],
                "champion": "xgboost|...", "start_pr_auc": 0.30, "end_pr_auc": 0.32}

    def services(self):
        return Services(self.data_review, self.evaluate, self.arena, self.improve, lambda t, g: self.lessons.append((t, g)))


QUIET = {"log": lambda _m: None}


def test_planner_builds_different_plans_for_each_depth():
    planner = MissionPlanner()
    quick = planner.plan(Mission(goal="improve it", depth="quick"))
    standard = planner.plan(Mission(goal="improve it", depth="standard"))
    assert quick.names() == ["data_review", "evaluate", "improve", "reflect"]
    assert standard.names() == ["data_review", "evaluate", "arena", "improve", "reflect"]
    assert standard.step("improve").params["iterations"] == 4


def test_mission_input_is_validated():
    with pytest.raises(ValueError):
        Mission(goal="x")
    with pytest.raises(ValueError):
        Mission(goal="a valid goal", depth="reckless")
    assert Mission(goal="a valid goal", depth="quick", improve_iterations=3).iterations == 3


def test_replan_cancels_modelling_when_the_data_is_broken():
    planner = MissionPlanner()
    plan = planner.plan(Mission(goal="improve it"))
    new = planner.replan(plan, Mission(goal="improve it"), {"verdict": "attention", "findings": [finding("critical", "Duplicate labels")]}, None)
    assert new.names() == ["data_review", "reflect"] and "Duplicate labels" in new.rationale


def test_replan_halves_the_budget_for_a_healthy_model_and_passes_ideas_otherwise():
    planner, mission = MissionPlanner(), Mission(goal="improve it")
    plan = planner.plan(mission)
    healthy = planner.replan(plan, mission, {"verdict": "healthy"}, {"verdict": "healthy", "hypotheses": []})
    assert healthy.step("improve").params["iterations"] == 2
    ideas = planner.replan(plan, mission, {"verdict": "watch"}, {"verdict": "watch", "hypotheses": ["a", "b"]})
    assert ideas.step("improve").params["focus"] == ["a", "b"] and ideas.step("improve").params["iterations"] == 4


def test_a_full_mission_runs_every_step_in_order_and_remembers_lessons():
    fake = Fake(adopt_rounds=())
    state = run_mission(Mission(goal="improve fraud detection", depth="standard", max_rounds=1), fake.services(), **QUIET)
    assert fake.calls == ["data_review", "evaluate", "arena", "improve:4:1"]
    assert [t["step"] for t in state["timeline"]] == ["plan", "data_review", "evaluate", "arena", "improve", "reflect"]
    assert state["reflection"]["lessons"] and fake.lessons
    assert all(t["status"] == "ok" for t in state["timeline"])


def test_a_second_round_runs_only_if_the_first_adopted_something():
    again = Fake(adopt_rounds=(1,))
    state = run_mission(Mission(goal="improve fraud detection", depth="standard", max_rounds=3), again.services(), **QUIET)
    assert state["round"] == 2 and again.calls.count("improve:4:1") == 2  # round 2 adopted nothing, so it stops
    once = Fake(adopt_rounds=())
    assert run_mission(Mission(goal="improve fraud detection", depth="standard", max_rounds=3), once.services(), **QUIET)["round"] == 1


def test_the_round_limit_is_respected_even_if_every_round_adopts():
    fake = Fake(adopt_rounds=(1, 2, 3, 4))
    state = run_mission(Mission(goal="improve fraud detection", depth="quick", max_rounds=2), fake.services(), **QUIET)
    assert state["round"] == 2


def test_critical_data_problems_stop_the_mission_before_any_modelling():
    fake = Fake(data_verdict="attention")
    state = run_mission(Mission(goal="improve fraud detection"), fake.services(), **QUIET)
    assert fake.calls == ["data_review"] and "Duplicate labels" in state["halted"]
    assert [t["step"] for t in state["timeline"]][-1] == "reflect"
    assert "Halted" in mission_markdown(state)


def test_a_failing_step_is_recorded_and_the_mission_continues():
    fake = Fake(fail="arena")
    state = run_mission(Mission(goal="improve fraud detection"), fake.services(), **QUIET)
    assert any("arena" in e and "exploded" in e for e in state["errors"])
    assert "improve:4:1" in fake.calls and state["timeline"][-1]["step"] == "reflect"
    assert any(t["status"].startswith("failed") for t in state["timeline"])


def test_the_budget_and_kill_switch_stop_the_mission_but_still_reflect(tmp_path):
    fake = Fake()
    budget = Budget(kill_switch_path=tmp_path / "STOP")
    (tmp_path / "STOP").write_text("")
    state = run_mission(Mission(goal="improve fraud detection"), fake.services(), budget=budget, **QUIET)
    assert state["halted"].startswith("stopped") and fake.calls == []
    assert state["timeline"][-1]["step"] == "reflect"
    with pytest.raises(BudgetExceeded):
        b = Budget(max_experiments=0)
        b.charge_experiment()
        b.check()


def test_reflection_writes_grounded_lessons_and_decides_whether_to_go_again():
    results = {"data_review": {"verdict": "watch", "findings": [finding("warning", "risk_events: low coverage")]},
               "evaluate": {"findings": [finding("warning", "Weaker on region=east")], "hypotheses": ["Try X"]},
               "arena": {"winner": "xgboost", "rationale": "tied; cheapest"}}
    history = [{"hypothesis": "deeper", "verdict": "adopt", "reason": "+0.02 [+0.01, +0.03]"},
               {"hypothesis": "wider", "verdict": "reject", "reason": "interval includes zero"}]
    r = reflect(results, history, round_number=1, max_rounds=2, last_round_adopted=1)
    texts = " | ".join(t for t, _ in r.lessons)
    assert "risk_events: low coverage" in texts and "region=east" in texts and "Judge served xgboost" in texts
    assert "Mission adopted: deeper" in texts and "Mission rejected: wider" in texts
    assert r.again and r.adopted == 1 and r.next_steps[0].startswith("The last round found")
    assert not reflect(results, history, 2, 2, 1).again and not reflect(results, history, 1, 2, 0).again


def test_the_report_summarises_the_mission():
    fake = Fake()
    state = run_mission(Mission(goal="improve fraud detection", depth="quick", max_rounds=1), fake.services(), **QUIET)
    payload = mission_payload(state)
    assert payload["start_pr_auc"] == 0.3 and payload["end_pr_auc"] == 0.32 and payload["rounds"] == 1
    text = mission_markdown(state)
    for part in ("# Mission report", "## Plan", "## Evaluation", "## Improvement", "## Reflection", "sealed test split was not used"):
        assert part in text


def test_plan_model_round_trips():
    plan = MissionPlanner().plan(Mission(goal="improve fraud detection"))
    assert Plan(**plan.model_dump()).names() == plan.names()


def test_two_round_mission_keeps_the_first_round_start_score():
    class TwoRound(Fake):
        def improve(self, iterations, focus):
            out = super().improve(iterations, focus)
            out["start_pr_auc"], out["end_pr_auc"] = (0.30, 0.32) if self.round == 1 else (0.32, 0.33)
            return out

    fake = TwoRound(adopt_rounds=(1, 2))
    state = run_mission(Mission(goal="improve it", depth="standard", max_rounds=2), fake.services(), **QUIET)
    payload = mission_payload(state)
    assert payload["start_pr_auc"] == 0.30 and payload["end_pr_auc"] == 0.33
