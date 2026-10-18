"""The arena: every registered algorithm must build, fit, score and stay inside its bounds."""

import numpy as np
import pandas as pd
import pytest
import yaml

from agents.spec import PARAM_BOUNDS, ExperimentSpec
from arena.estimators import IsolationForestScorer
from arena.registry import REGISTRY, build_pipeline, get_family, names
from arena.runner import (
    ArenaTask,
    markdown_report,
    paired_pr_auc_draws,
    rank_results,
    train_candidate,
)
from models.training import CONFIG_PATH

NUMERIC = ["usage__calls__7d", "usage__calls__30d", "risk__events__30d"]
CATEGORICAL = ["customer__plan"]
# small and fast: the point is that each algorithm runs, not that it is accurate
FAST = {
    "logistic_regression": {"C": 0.1, "max_iter": 200},
    "random_forest": {"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 5, "max_features": 0.5},
    "extra_trees": {"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 5, "max_features": 0.5},
    "xgboost": {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.2},
    "lightgbm": {"n_estimators": 20, "num_leaves": 6, "learning_rate": 0.2, "min_child_samples": 10},
    "hist_gradient_boosting": {"max_iter": 20, "max_depth": 3, "learning_rate": 0.2, "min_samples_leaf": 10},
    "catboost": {"iterations": 20, "depth": 3, "learning_rate": 0.2},
    "isolation_forest": {"n_estimators": 30, "max_samples": 128, "max_features": 1.0},
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    n = 1500
    frame = pd.DataFrame(
        {
            "usage__calls__7d": rng.normal(size=n),
            "usage__calls__30d": rng.normal(size=n),
            "risk__events__30d": rng.normal(size=n),
            "customer__plan": rng.choice(["prepaid", "postpaid", "business"], n),
        }
    )
    frame.loc[rng.random(n) < 0.1, "usage__calls__30d"] = np.nan  # missing values are normal here
    logit = 2.5 * frame["usage__calls__7d"] + (frame["customer__plan"] == "prepaid") - 3.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit.to_numpy()))).astype(int)
    return frame.iloc[:1000], y[:1000], frame.iloc[1000:], y[1000:]


def test_fast_params_cover_every_registered_model():
    assert set(FAST) == set(names())
    assert len(names()) == 8


@pytest.mark.parametrize("name", names())
def test_every_model_fits_and_scores_with_missing_values(name, data):
    train_x, train_y, val_x, val_y = data
    pipeline = build_pipeline(name, FAST[name], NUMERIC, CATEGORICAL, 1)
    pipeline.fit(train_x, train_y)
    probs = pipeline.predict_proba(val_x)[:, 1]
    assert probs.shape == (len(val_x),)
    assert np.all((probs >= 0) & (probs <= 1)) and np.isfinite(probs).all()
    assert len(np.unique(probs)) > 5, "scores should rank rows, not collapse to a few values"


@pytest.mark.parametrize("name", [n for n in names() if n != "isolation_forest"])
def test_supervised_models_beat_random_on_a_learnable_signal(name, data):
    train_x, train_y, val_x, val_y = data
    task = ArenaTask(name, FAST[name], NUMERIC, CATEGORICAL, 1)
    result = train_candidate(task, train_x, train_y, val_x, val_y, 0.05)
    assert result.ok, result.error
    assert result.metrics["pr_auc"] > val_y.mean() * 1.5
    assert result.size_mb > 0 and result.train_seconds >= 0


def test_a_failing_candidate_becomes_a_result_not_a_crash(data):
    train_x, train_y, val_x, val_y = data
    task = ArenaTask("logistic_regression", {"not_a_parameter": 1}, NUMERIC, CATEGORICAL, 1)
    result = train_candidate(task, train_x, train_y, val_x, val_y, 0.05)
    assert result.ok is False and result.error


def test_isolation_forest_scores_outliers_higher_without_labels():
    rng = np.random.default_rng(0)
    inliers = rng.normal(size=(500, 4))
    outliers = rng.normal(loc=8.0, size=(10, 4))
    scorer = IsolationForestScorer(n_estimators=50, max_samples=128, random_state=0).fit(inliers)
    score_in = scorer.predict_proba(inliers[:50])[:, 1].mean()
    score_out = scorer.predict_proba(outliers)[:, 1].mean()
    assert score_out > 0.9 > score_in


def test_identical_models_are_reported_as_tied(data):
    train_x, train_y, val_x, val_y = data
    task = ArenaTask("xgboost", FAST["xgboost"], NUMERIC, CATEGORICAL, 1)
    first = train_candidate(task, train_x, train_y, val_x, val_y, 0.05)
    second = train_candidate(task, train_x, train_y, val_x, val_y, 0.05)
    second.model = "lightgbm"  # same scores under another name
    rows = rank_results([first, second], val_y, n_boot=30, seed=1)
    assert [r["status"] for r in rows] == ["best", "tied with best"]
    assert rows[1]["diff_vs_best"] == 0.0


def test_a_clearly_worse_model_is_ranked_worse(data):
    _, _, _, val_y = data
    rng = np.random.default_rng(3)
    from arena.runner import ArenaResult

    good = ArenaResult("xgboost", True, val_probs=val_y * 0.8 + rng.random(len(val_y)) * 0.2,
                       metrics={"pr_auc": 0.95, "roc_auc": 0.99, "precision_at_review": 1.0, "recall_at_review": 0.5})
    bad = ArenaResult("lightgbm", True, val_probs=rng.random(len(val_y)),
                      metrics={"pr_auc": 0.1, "roc_auc": 0.5, "precision_at_review": 0.1, "recall_at_review": 0.1})
    rows = rank_results([bad, good], val_y, n_boot=30, seed=1)
    assert rows[0]["model"] == "xgboost" and rows[0]["status"] == "best"
    assert rows[1]["status"] == "worse"
    assert "worse" in markdown_report(rows, 0.05, len(val_y), int(val_y.sum()))


def test_bootstrap_draws_are_paired_and_reproducible(data):
    _, _, _, val_y = data
    p = np.random.default_rng(1).random(len(val_y))
    a = paired_pr_auc_draws(val_y, {"m": p, "n": p}, 20, seed=5)
    b = paired_pr_auc_draws(val_y, {"m": p}, 20, seed=5)
    assert np.array_equal(a["m"], a["n"]) and np.array_equal(a["m"], b["m"])


# ---- configuration and the experiment spec ------------------------------------------------
def test_every_registered_model_has_a_config_block_with_in_bounds_values():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert set(config["models"]) == set(names())
    for name, params in config["models"].items():
        for key, value in params.items():
            if key in PARAM_BOUNDS[name]:
                low, high, _ = PARAM_BOUNDS[name][key]
                assert low <= value <= high, f"{name}.{key}={value} outside [{low}, {high}]"


def test_candidates_and_arena_lists_only_name_registered_models():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert set(config["training"]["candidates"]) <= set(names())
    assert set(config["arena"]["models"]) <= set(names())


def test_config_params_are_accepted_by_the_estimators():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    for name, params in config["models"].items():
        assert build_pipeline(name, params, NUMERIC, CATEGORICAL, 1) is not None


def test_spec_accepts_every_registered_model_and_rejects_unknown_ones():
    for name in names():
        spec = ExperimentSpec(model=name, groups=["usage"], windows=[30])
        assert spec.problems(["usage"], [30]) == []
    with pytest.raises(ValueError, match="unknown model"):
        ExperimentSpec(model="neural_net", groups=["usage"], windows=[30])


def test_spec_bounds_are_enforced_per_model():
    ok = ExperimentSpec(model="lightgbm", groups=["usage"], windows=[30], params={"num_leaves": 16})
    assert ok.problems(["usage"], [30]) == []
    too_big = ExperimentSpec(model="lightgbm", groups=["usage"], windows=[30], params={"num_leaves": 500})
    assert too_big.problems(["usage"], [30])
    wrong_model = ExperimentSpec(model="lightgbm", groups=["usage"], windows=[30], params={"depth": 4})
    assert wrong_model.problems(["usage"], [30])  # depth belongs to catboost


def test_registry_families_declare_a_known_kind_and_preprocessing():
    for family in REGISTRY.values():
        assert family.kind in {"linear", "bagging", "boosting", "anomaly"}
        assert family.preprocessing in {"scaled", "imputed", "native"}
    assert get_family("catboost").kind == "boosting"
    with pytest.raises(ValueError):
        get_family("nope")
