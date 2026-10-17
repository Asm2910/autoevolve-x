import numpy as np
import pandas as pd
import pytest

from agents.spec import ExperimentSpec
from memory.store import MemoryStore
from models.training import LABEL
from scripts.confirm_champion import KIND, existing_confirmation, report_markdown
from tools.final_evaluation import conclusion_text, final_evaluation

CONFIG = {
    "training": {
        "seed": 1, "review_fraction": 0.05, "bootstrap_samples": 40,
        "threshold_grid": {"start": 0.05, "stop": 0.95, "step": 0.05},
    },
    "features": {"groups": ["usage", "customer"], "windows": [30]},
    "models": {
        "xgboost": {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.2},
        "logistic_regression": {"C": 1.0, "max_iter": 200},
    },
}
MANIFEST = {"features": ["usage__calls__30d", "customer__age"], "categorical": []}


def make_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    usage = rng.normal(size=n)
    age = rng.normal(size=n)
    label = (rng.random(n) < 1 / (1 + np.exp(-(2.5 * usage - 2.5)))).astype(int)
    return pd.DataFrame({"usage__calls__30d": usage, "customer__age": age, LABEL: label})


@pytest.fixture(scope="module")
def result():
    champion = ExperimentSpec(model="xgboost", groups=["usage", "customer"], windows=[30])
    return final_evaluation(CONFIG, champion, MANIFEST, make_frame(3000, 1), make_frame(2000, 2), make_frame(2000, 3))


def test_result_has_champion_baseline_and_paired_difference(result):
    assert result["champion"]["spec"]["model"] == "xgboost"
    assert result["baseline"]["spec"]["model"] == "logistic_regression"
    low, high = result["diff_pr_auc"]["ci95"]
    assert low <= result["diff_pr_auc"]["mean"] <= high
    assert result["n_test_rows"] == 2000
    assert result["verdict"] in {"confirmed", "not_confirmed"}
    assert (result["verdict"] == "confirmed") == (low > 0)


def test_thresholds_come_from_validation_grid(result):
    grid = np.round(np.arange(0.05, 0.95 + 1e-9, 0.05), 4)
    assert result["champion"]["threshold"] in grid


def test_text_outputs_mention_both_models(result):
    conclusion = conclusion_text(result)
    assert "Champion xgboost" in conclusion and "baseline" in conclusion
    report = report_markdown(result, conclusion)
    assert "champion (xgboost)" in report and "baseline (logistic_regression)" in report


def test_confirmation_is_refused_the_second_time(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    assert existing_confirmation(store, "data-1") is None
    store.record_experiment(KIND, "champion", {"a": 1}, {}, dataset_sha256="data-1")
    assert existing_confirmation(store, "data-1") is not None
    assert existing_confirmation(store, "data-2") is None  # a new dataset may be confirmed once too