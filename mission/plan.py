"""Mission and plan: what was asked, and the ordered steps chosen to do it.

The planner is rule-based on purpose. A plan is a list of steps from a fixed menu, so a mistake in
planning can only choose a bad order of safe steps, never invent an action. The planner looks at the
requested depth first, then re-plans once it has seen the data review and the baseline evaluation:

  - if the data review found a critical problem, modelling is cancelled and the mission reports why;
  - if the evaluation is healthy and shows nothing to fix, the improvement budget is cut;
  - otherwise the evaluation's ideas are handed to the improvement step as its focus.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

StepName = Literal["data_review", "evaluate", "arena", "improve", "reflect"]
DEPTHS = {
    # depth: (improvement experiments per round, rounds, whether to run the model arena)
    "quick": (2, 1, False),
    "standard": (4, 2, True),
    "thorough": (6, 3, True),
}


class Mission(BaseModel):
    goal: str = Field(min_length=3, max_length=300)
    depth: Literal["quick", "standard", "thorough"] = "standard"
    max_rounds: int | None = Field(default=None, ge=1, le=5)
    improve_iterations: int | None = Field(default=None, ge=1, le=20)

    @property
    def rounds(self) -> int:
        return self.max_rounds or DEPTHS[self.depth][1]

    @property
    def iterations(self) -> int:
        return self.improve_iterations or DEPTHS[self.depth][0]


class PlanStep(BaseModel):
    name: StepName
    why: str
    params: dict = Field(default_factory=dict)


class Plan(BaseModel):
    steps: list[PlanStep]
    rationale: str

    def names(self) -> list[str]:
        return [s.name for s in self.steps]

    def step(self, name: str) -> PlanStep | None:
        return next((s for s in self.steps if s.name == name), None)


class MissionPlanner:
    def plan(self, mission: Mission) -> Plan:
        arena = DEPTHS[mission.depth][2]
        steps = [
            PlanStep(name="data_review", why="Check the sources and features before trusting any model."),
            PlanStep(name="evaluate", why="Measure the current model and find where it is weak."),
        ]
        if arena:
            steps.append(PlanStep(name="arena", why="Compare the candidate algorithms and let the judge pick one to serve."))
        steps.append(PlanStep(name="improve", why="Let the agent test changes with a fixed statistical rule.",
                              params={"iterations": mission.iterations}))
        steps.append(PlanStep(name="reflect", why="Turn the results into verified lessons and next steps."))
        return Plan(steps=steps, rationale=f"{mission.depth} mission: {len(steps)} steps, up to {mission.rounds} improvement round(s).")

    def replan(self, plan: Plan, mission: Mission, data: dict | None, evaluation: dict | None) -> Plan:
        """Adjust the plan using what the first two steps found."""
        notes = []
        if data and data.get("verdict") == "attention":
            kept = [s for s in plan.steps if s.name in {"data_review", "reflect"}]
            critical = [f["title"] for f in data.get("findings", []) if f.get("severity") == "critical"][:3]
            return Plan(steps=kept, rationale="Stopped before modelling: the data review found critical problems: "
                        + "; ".join(critical or ["see report"]) + ".")
        steps = []
        for step in plan.steps:
            if step.name == "improve":
                params = dict(step.params)
                ideas = list((evaluation or {}).get("hypotheses", [])) + list((data or {}).get("hypotheses", []))
                if evaluation and evaluation.get("verdict") == "healthy" and not ideas:
                    params["iterations"] = max(1, params["iterations"] // 2)
                    notes.append("evaluation is healthy with nothing to fix, so the improvement budget is halved")
                elif ideas:
                    notes.append(f"{len(ideas)} idea(s) from the swarms are passed to the improvement step")
                params["focus"] = ideas[:6]
                step = step.model_copy(update={"params": params})
            steps.append(step)
        return Plan(steps=steps, rationale=" ".join([plan.rationale, *(n[0].upper() + n[1:] + "." for n in notes)]))
