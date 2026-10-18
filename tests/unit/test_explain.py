import numpy as np
import pandas as pd
import pytest

from models.explain import contributions, supports_explanations, top_reasons
from models.training import build_pipeline

PARAMS = {
    "xgboost": {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.2},
    "lightgbm": {"n_estimators": 30, "num_leaves": 8, "learning_rate": 0.2, "min_child_samples": 10},
    "catboost": {"iterations": 30, "depth": 3, "learning_rate": 0.2},
}


@pytest.fixture(scope="module", params=list(PARAMS))
def fitted(request):
    rng = np.random.default_rng(0)
    n = 600
    frame = pd.DataFrame(
        {
            "usage__calls__7d": rng.normal(size=n),
            "usage__calls__30d": rng.normal(size=n),
            "customer__plan": rng.choice(["prepaid", "postpaid", "business"], n),
        }
    )
    logit = 2.0 * frame["usage__calls__7d"] + (frame["customer__plan"] == "prepaid") - 2.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    pipeline = build_pipeline(
        request.param, PARAMS[request.param], ["usage__calls__7d", "usage__calls__30d"], ["customer__plan"], 1
    )
    pipeline.fit(frame, y)
    return pipeline, frame


def test_contributions_are_additive(fitted):
    pipeline, frame = fitted
    per_feature, bias = contributions(pipeline, frame)
    prob = np.clip(pipeline.predict_proba(frame)[:, 1], 1e-9, 1 - 1e-9)
    logit = np.log(prob / (1 - prob))
    assert np.allclose(bias + per_feature.sum(axis=1), logit, atol=1e-3)
    assert list(per_feature.columns) == list(frame.columns)  # one-hot summed back


def test_top_reasons_name_the_driving_feature(fitted):
    pipeline, frame = fitted
    high = frame.assign(usage__calls__7d=3.0).iloc[[0]]
    reasons = top_reasons(pipeline, high, top_k=3)[0]
    assert reasons and reasons[0]["feature"] == "usage__calls__7d"
    assert all(r["contribution"] > 0 for r in reasons)


def test_logistic_regression_not_supported():
    pipeline = build_pipeline("logistic_regression", {"C": 1.0}, ["a"], ["b"], 1)
    assert supports_explanations(pipeline) is False


@pytest.mark.parametrize("name", ["random_forest", "extra_trees", "hist_gradient_boosting", "isolation_forest"])
def test_models_without_exact_contributions_are_served_without_explanations(name):
    params = {
        "random_forest": {"n_estimators": 5},
        "extra_trees": {"n_estimators": 5},
        "hist_gradient_boosting": {"max_iter": 5},
        "isolation_forest": {"n_estimators": 5},
    }[name]
    assert supports_explanations(build_pipeline(name, params, ["a"], ["b"], 1)) is False
