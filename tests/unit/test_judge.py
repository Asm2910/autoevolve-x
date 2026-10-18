"""The judge must follow its written policy: constraints, utility, statistical ties, cheapest wins."""

import numpy as np
import pytest

from judge.utility import (
    EXPLAINABLE,
    Candidate,
    JudgeAgent,
    JudgeRequest,
    UtilityConfig,
    judge,
    review_depth,
    utility_per_10k,
)
from models.explain import EXPLAINABLE_LIBRARIES

N = 20000
CFG = UtilityConfig()


@pytest.fixture(scope="module")
def y():
    return (np.random.default_rng(0).random(N) < 0.02).astype(int)


def scores(y, strength, seed):
    return y * strength + np.random.default_rng(seed).normal(size=N)


def test_utility_matches_hand_arithmetic():
    # 100 reviewed of 10,000, 40 of them fraud: 40*400 - 100*6 - 60*3 = 15,220 per 10,000 accounts
    assert utility_per_10k(40, 100, 10_000, CFG) == pytest.approx(15_220)
    assert utility_per_10k(0, 100, 10_000, CFG) == pytest.approx(-900)
    assert review_depth(46_589, 0.01) == 466 and review_depth(10, 0.01) == 1


def test_explainable_set_matches_what_the_explainer_supports():
    assert set(EXPLAINABLE) == set(EXPLAINABLE_LIBRARIES)


def test_a_clearly_better_model_wins_even_if_it_is_slower(y):
    good = Candidate("xgboost", scores(y, 3.0, 1), train_seconds=300, size_mb=5)
    weak = Candidate("lightgbm", scores(y, 1.0, 2), train_seconds=1, size_mb=0.1)
    verdict = judge([good, weak], y, CFG, 0.01, n_boot=100)
    assert verdict.winner == "xgboost"
    assert any(r["model"] == "lightgbm" and r["status"] == "worse" for r in verdict.rows)


def test_tied_models_are_decided_by_cost_not_by_a_tiny_score_edge(y):
    base = scores(y, 2.5, 7)
    slow = Candidate("catboost", base + 1e-6, train_seconds=100, size_mb=0.2)  # best by a hair, slowest
    fast = Candidate("xgboost", base, train_seconds=5, size_mb=0.4)
    verdict = judge([slow, fast], y, CFG, 0.01, n_boot=100)
    assert verdict.winner == "xgboost" and "tied" in verdict.rationale


def test_models_that_cannot_explain_are_excluded_when_explanations_are_required(y):
    forest = Candidate("random_forest", scores(y, 4.0, 3), train_seconds=1, size_mb=1)  # best, cannot explain
    boost = Candidate("lightgbm", scores(y, 2.5, 4), train_seconds=9, size_mb=1)
    verdict = judge([forest, boost], y, CFG, 0.01, n_boot=100)
    assert verdict.winner == "lightgbm"
    assert next(r for r in verdict.rows if r["model"] == "random_forest")["status"] == "excluded"
    relaxed = judge([forest, boost], y, UtilityConfig(require_explanations=False), 0.01, n_boot=100)
    assert relaxed.winner == "random_forest"


def test_if_nothing_can_explain_the_requirement_is_waived_and_the_report_says_so(y):
    only = [Candidate("random_forest", scores(y, 3.0, 5)), Candidate("logistic_regression", scores(y, 1.0, 6))]
    verdict = judge(only, y, CFG, 0.01, n_boot=60)
    assert verdict.winner == "random_forest" and "waived" in verdict.rationale


def test_a_tied_model_far_below_the_best_is_not_chosen(y):
    best = Candidate("catboost", scores(y, 2.0, 8), train_seconds=50, size_mb=1)
    cheap = Candidate("xgboost", scores(y, 1.6, 9), train_seconds=1, size_mb=1)
    verdict = judge([best, cheap], y, UtilityConfig(max_point_loss=0.01), 0.01, n_boot=100)
    assert verdict.winner == "catboost"


def test_the_verdict_is_reproducible_and_reportable(y):
    cands = [Candidate("xgboost", scores(y, 2.5, 1), 5, 0.4), Candidate("lightgbm", scores(y, 2.4, 2), 9, 0.3)]
    a, b = judge(cands, y, CFG, 0.01, n_boot=80, seed=3), judge(cands, y, CFG, 0.01, n_boot=80, seed=3)
    assert a.to_dict() == b.to_dict()
    text = a.markdown()
    assert "Served model:" in text and "SERVED" in text and "per 10,000" in text


def test_config_reads_the_judge_block_and_ignores_unknown_keys():
    cfg = UtilityConfig.from_config({"judge": {"fraud_value": 100, "nonsense": 1}})
    assert cfg.fraud_value == 100 and cfg.review_cost == UtilityConfig().review_cost
    assert UtilityConfig.from_config({}) == UtilityConfig()


def test_no_candidates_is_an_error(y):
    with pytest.raises(ValueError):
        judge([], y, CFG, 0.01)


def test_judge_agent_runs_under_the_harness_and_keeps_its_verdict(y):
    cands = [Candidate("xgboost", scores(y, 2.5, 1), 5, 0.4)]
    agent = JudgeAgent(cands, y, CFG, 0.01)
    out = agent.run(JudgeRequest(n_boot=40))
    assert out.winner == "xgboost" and agent.verdict is not None
