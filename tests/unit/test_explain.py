import numpy as np
import pandas as pd
import pytest

from models.explain import contributions, supports_explanations, top_reasons
from models.training import build_pipeline


@pytest.fixture(scope="module")
def fitted():
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
    params = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.2}
    pipeline = build_pipeline(
        "xgboost", params, ["usage__calls__7d", "usage__calls__30d"], ["customer__plan"], 1
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
