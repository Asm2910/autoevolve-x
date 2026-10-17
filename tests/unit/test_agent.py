import pytest

from agents.decision import decide
from agents.graph import build_agent, load_champion
from agents.proposers import RuleProposer
from agents.spec import ExperimentSpec
from memory.store import MemoryStore

GROUPS = ["customer", "usage", "recharge", "network", "risk"]
WINDOWS = [7, 30, 90]
BASE_XGB = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "colsample_bytree": 0.5, "min_child_weight": 5}
CONFIG = {
    "features": {"groups": GROUPS, "windows": WINDOWS},
    "agent": {"min_gain": 0.01},
    "tracking": {"enabled": False},
}


def spec(**changes) -> ExperimentSpec:
    base = {"model": "xgboost", "groups": GROUPS, "windows": WINDOWS, "hypothesis": "baseline"}
    return ExperimentSpec(**{**base, **changes})


# ---- spec ------------------------------------------------------------------
def test_spec_accepts_valid_and_rejects_unknown_things():
    assert spec().problems(GROUPS, WINDOWS) == []
    assert spec(groups=["usage", "magic"]).problems(GROUPS, WINDOWS)
    assert spec(windows=[7, 365]).problems(GROUPS, WINDOWS)
    assert spec(params={"max_depth": 50}).problems(GROUPS, WINDOWS)
    assert spec(params={"gamma": 1}).problems(GROUPS, WINDOWS)


def test_spec_key_ignores_hypothesis_and_order():
    a = spec(groups=["usage", "risk"], hypothesis="one")
    b = spec(groups=["risk", "usage"], hypothesis="two")
    assert a.key() == b.key()
    assert a.key() != spec(groups=["usage"]).key()


def test_resolved_params_keep_integers_as_integers():
    resolved = spec(params={"max_depth": 4.0, "learning_rate": 0.1}).resolved_params(BASE_XGB)
    assert resolved["max_depth"] == 4 and isinstance(resolved["max_depth"], int)
    assert resolved["learning_rate"] == 0.1
    assert resolved["n_estimators"] == 300


def test_spec_forbids_extra_fields():
    with pytest.raises(ValueError):
        ExperimentSpec(model="xgboost", groups=["usage"], windows=[7], shell="rm -rf /")


# ---- decision rule -----------------------------------------------------------
@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"diff_mean": 0.03, "diff_ci": [0.01, 0.05]}, "adopt"),
        ({"diff_mean": 0.005, "diff_ci": [0.001, 0.009]}, "reject"),  # significant but tiny
        ({"diff_mean": 0.02, "diff_ci": [-0.01, 0.05]}, "reject"),  # interval includes zero
        ({"diff_mean": -0.05, "diff_ci": [-0.08, -0.02]}, "reject"),  # worse
    ],
)
def test_decision_rule(result, expected):
    verdict, reason = decide(result, min_gain=0.01)
    assert verdict == expected
    assert reason


# ---- proposer ----------------------------------------------------------------
def test_rule_proposer_skips_tried_and_runs_out():
    proposer = RuleProposer(BASE_XGB)
    champion = spec()
    tried: list[str] = []
    seen = []
    while (candidate := proposer.propose(champion, {"tried_keys": tried})) is not None:
        assert candidate.problems(GROUPS, WINDOWS) == []
        assert candidate.key() not in tried
        tried.append(candidate.key())
        seen.append(candidate)
        assert len(seen) < 50
    assert len(seen) >= 8


def test_rule_proposer_stays_within_bounds():
    deep = spec(params={"max_depth": 8, "learning_rate": 0.01, "n_estimators": 600})
    for candidate in RuleProposer(BASE_XGB).candidates(deep):
        assert candidate.problems(GROUPS, WINDOWS) == []


# ---- the graph, with a fake runner (no training) -------------------------------
class FakeRunner:
    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.ran: list[str] = []
        self.champion = None

    def baseline_spec(self):
        return spec()

    def set_champion(self, champion):
        self.champion = champion

    def run(self, candidate):
        self.ran.append(candidate.key())
        mean, low, high = self.outcomes.get(candidate.key(), (0.0, -0.02, 0.02))
        return {"val_pr_auc": 0.3 + mean, "champion_val_pr_auc": 0.3, "diff_mean": mean, "diff_ci": [low, high],
                "n_features": 10}


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory.db")


def invoke(agent, champion, iterations):
    return agent.invoke(
        {"iteration": 0, "max_iterations": iterations, "retries": 0,
         "champion": {**champion.identity(), "hypothesis": champion.hypothesis}, "history": []},
        config={"recursion_limit": 100},
    )


def test_agent_runs_records_and_adopts(store):
    proposer = RuleProposer(BASE_XGB)
    first = proposer.propose(spec(), {"tried_keys": []})
    runner = FakeRunner({first.key(): (0.04, 0.02, 0.06)})
    agent = build_agent(runner, store, proposer, CONFIG, "data-1", log=lambda _m: None)

    state = invoke(agent, spec(), 3)

    assert len(state["history"]) == 3
    assert state["history"][0]["verdict"] == "adopt"
    assert runner.champion.key() == first.key()
    assert len(store.recent("agent_experiment")) == 3
    assert any("Adopted" in lesson["text"] for lesson in store.lessons())
    assert load_champion(store, runner, "data-1").key() == first.key()
    assert load_champion(store, runner, "other-data").key() == spec().key()


def test_agent_never_repeats_an_experiment_across_runs(store):
    proposer = RuleProposer(BASE_XGB)
    runner = FakeRunner()
    agent = build_agent(runner, store, proposer, CONFIG, "data-1", log=lambda _m: None)
    invoke(agent, spec(), 3)
    invoke(agent, spec(), 3)
    assert len(runner.ran) == 6
    assert len(set(runner.ran)) == 6


def test_agent_stops_when_proposer_is_exhausted(store):
    class OneShot:
        done = False

        def propose(self, champion, context):
            if self.done:
                return None
            self.done = True
            return spec(params={"max_depth": 4})

    agent = build_agent(FakeRunner(), store, OneShot(), CONFIG, "d", log=lambda _m: None)
    state = invoke(agent, spec(), 5)
    assert len(state["history"]) == 1
    assert "nothing left" in state["stop_reason"]


def test_invalid_proposals_never_reach_the_runner(store):
    class Reckless:
        def propose(self, champion, context):
            return spec(params={"max_depth": 999}, groups=["usage", "secrets"])

    runner = FakeRunner()
    agent = build_agent(runner, store, Reckless(), CONFIG, "d", log=lambda _m: None)
    state = invoke(agent, spec(), 3)
    assert runner.ran == []
    assert state.get("history") == []


def test_duplicate_proposals_are_skipped_by_memory(store):
    candidate = spec(params={"max_depth": 4})

    class Stubborn:
        calls = 0

        def propose(self, champion, context):
            self.calls += 1
            return candidate if self.calls <= 3 else None

    runner = FakeRunner()
    agent = build_agent(runner, store, Stubborn(), CONFIG, "d", log=lambda _m: None)
    invoke(agent, spec(), 5)
    assert runner.ran == [candidate.key()]  # proposed three times, run once