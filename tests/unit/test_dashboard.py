from pathlib import Path

import pytest

from dashboard import charts, data
from memory.store import MemoryStore

APP = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"

FINAL = {
    "created_at": "2026-10-17T09:36:57+00:00", "dataset_sha256": "ab" * 32, "git_commit": "2b55d32", "git_dirty": 0,
    "metrics": {
        "champion": {"spec": {"model": "xgboost"}, "test": {"pr_auc": 0.272}, "test_ci95": {"pr_auc": [0.225, 0.324]}},
        "baseline": {"spec": {"model": "logistic_regression"}, "test": {"pr_auc": 0.211},
                     "test_ci95": {"pr_auc": [0.17, 0.25]}},
        "diff_pr_auc": {"mean": 0.061, "ci95": [0.030, 0.094]},
        "n_test_rows": 46252, "n_test_positives": 399, "verdict": "confirmed",
    },
}


def entry(i, mean, low, high, verdict="reject", hypothesis="Try deeper trees"):
    return {"id": i, "hypothesis": hypothesis, "verdict": verdict, "conclusion": "change: interval includes zero",
            "mean": mean, "low": low, "high": high}


def test_signed_uses_a_real_minus_sign():
    assert charts.signed(0.0612) == "+0.061"
    assert charts.signed(-0.021) == "\u22120.021"
    assert charts.signed(0.0) == "0.000"


def test_pos_is_clamped_to_the_axis():
    assert charts.pos(0.0, -0.04, 0.04) == 50
    assert charts.pos(9.0, -0.04, 0.04) == 100
    assert charts.pos(-9.0, -0.04, 0.04) == 0


def test_humanize_feature_names():
    assert data.humanize("usage__max_unique_destinations__30d") == "Max unique destinations, 30 days"
    assert data.humanize("customer__tenure_days") == "Tenure days"
    assert data.group_of("risk__avg_risk_score__7d") == "risk"


def test_masthead_counts_adoptions():
    html = charts.masthead([entry(1, 0.0, -0.01, 0.01), entry(2, 0.03, 0.02, 0.04, verdict="adopt")])
    assert "tried 2 ideas and kept 1" in html
    assert "not run any experiments" in charts.masthead([])


def test_ledger_shows_every_experiment_and_the_adopt_zone():
    html = charts.ledger_section([entry(1, 0.002, -0.01, 0.015), entry(2, 0.04, 0.02, 0.06, verdict="adopt")])
    assert html.count('class="lrow') == 2
    assert "Adopted." in html and "Rejected." in html
    assert "Adopt zone" in html
    assert "1 of 2" in html


def test_ledger_empty_state_tells_the_user_what_to_run():
    assert "scripts.run_agent" in charts.ledger_section([])


def test_ledger_escapes_model_written_text():
    html = charts.ledger_section([entry(1, 0.0, -0.01, 0.01, hypothesis="<script>alert(1)</script>")])
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_final_test_confirmed_and_not_confirmed_wording():
    assert "gap is real" in charts.final_test_section(FINAL)
    unproven = {**FINAL, "metrics": {**FINAL["metrics"], "verdict": "not_confirmed",
                                     "diff_pr_auc": {"mean": 0.01, "ci95": [-0.02, 0.04]}}}
    assert "not proven" in charts.final_test_section(unproven)
    assert "confirm_champion" in charts.final_test_section(None)


def test_drift_section_handles_missing_and_present_data():
    assert "monitor_drift" in charts.drift_section([], None)
    checks = [{"name": "2026-03-31", "verdict": "stable", "metrics": {"n_major": 0, "n_moderate": 0, "n_features": 254}}]
    html = charts.drift_section(checks, [{"feature": "customer__tenure_days", "psi": 0.076}])
    assert "Stable" in html and "Tenure days" in html and "noticeable" in html


def test_signal_section_orders_groups_and_colours_them():
    html = charts.signal_section([("usage", 0.6), ("risk", 0.4)], [{"feature": "usage__calls__7d", "share": 0.05}])
    assert html.index("Usage") < html.index("Risk")
    assert "#eb6834" in html


def test_importance_shares_sum_to_one():
    import pandas as pd

    table = pd.DataFrame({"group": ["usage", "usage", "risk"], "share": [0.3, 0.2, 0.1]})
    shares = dict(data.importance_by_group(table))
    assert pytest.approx(sum(shares.values())) == 1.0 and shares["usage"] > shares["risk"]


def test_agent_ledger_reads_memory_for_one_dataset(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    metrics = {"diff_mean": 0.01, "diff_ci_low": -0.01, "diff_ci_high": 0.03}
    store.record_experiment("agent_experiment", "a", {"x": 1}, metrics, hypothesis="one", verdict="reject",
                            dataset_sha256="d1")
    store.record_experiment("agent_experiment", "b", {"x": 2}, metrics, hypothesis="two", verdict="adopt",
                            dataset_sha256="d1")
    store.record_experiment("agent_experiment", "c", {"x": 3}, metrics, hypothesis="other data", dataset_sha256="d2")
    rows = data.agent_ledger(store, "d1")
    assert [r["hypothesis"] for r in rows] == ["one", "two"]  # oldest first, other dataset left out
    assert data.latest_final_test(store) is None


def test_the_app_starts_without_errors_on_an_empty_memory(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("AUTOEVOLVE_MEMORY_DB", str(tmp_path / "empty.db"))
    app = testing.AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception