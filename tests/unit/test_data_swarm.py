"""The data agents must catch planted data problems and stay quiet on clean data."""

import numpy as np
import pandas as pd
import pytest

from swarm.data import (
    AGENTS,
    DataContext,
    DataRequest,
    DataSwarm,
    FeatureAgent,
    QualityAgent,
    RelationshipAgent,
    SchemaAgent,
    TableSpec,
    benjamini_hochberg,
    telecom_specs,
)

REQ = DataRequest(sample_rows=10_000)
WINDOW = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31 23:59:59"))


def make_tables(seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"S{i}" for i in range(300)]
    customers = pd.DataFrame({"subscriber_id": ids, "region": rng.choice(["north", "south"], 300),
                              "tenure_days": rng.integers(10, 900, 300)})
    n = 3000
    events = pd.DataFrame({
        "event_id": [f"E{i}" for i in range(n)], "subscriber_id": rng.choice(ids, n),
        "timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 80 * 24 * 3600, n), unit="s"),
        "calls": rng.integers(0, 20, n)})
    labels = pd.DataFrame({"subscriber_id": ids * 2, "snapshot_date": [pd.Timestamp("2026-03-01")] * 300 + [pd.Timestamp("2026-03-15")] * 300,
                           "risk_label": rng.random(600) < 0.1})
    labels["risk_label"] = labels["risk_label"].astype(int)
    return {"customers": customers, "events": events, "labels": labels}


SPECS = {
    "customers": TableSpec(required=["subscriber_id", "region", "tenure_days"], id_col="subscriber_id", time_col=None,
                           nonnegative=["tenure_days"], categories={"region": {"north", "south"}}),
    "events": TableSpec(required=["event_id", "subscriber_id", "timestamp", "calls"], id_col="event_id",
                        time_col="timestamp", nonnegative=["calls"]),
    "labels": TableSpec(required=["subscriber_id", "snapshot_date", "risk_label"], time_col="snapshot_date"),
}


def ctx(tables=None, **kw) -> DataContext:
    return DataContext(tables=tables or make_tables(), specs=SPECS, label_table="labels", time_window=WINDOW,
                       lookback_days=30, **kw)


def titles(out):
    return [f.title for f in out.findings]


def test_every_agent_is_deterministic_with_a_contract():
    for cls in AGENTS:
        assert cls.contract.kind == "deterministic" and cls.contract.responsibility


def test_clean_data_gets_no_warnings_from_the_first_three_agents():
    for cls in (SchemaAgent, QualityAgent, RelationshipAgent):
        out = cls(ctx()).run(REQ)
        assert all(f.severity == "info" for f in out.findings), (cls.__name__, titles(out))


def test_schema_agent_finds_missing_columns_and_unknown_categories():
    t = make_tables()
    t["customers"] = t["customers"].drop(columns=["tenure_days"])
    t["customers"].loc[:10, "region"] = "mars"
    out = SchemaAgent(ctx(t)).run(REQ)
    assert "customers: columns missing" in titles(out) and "customers.region: unexpected categories" in titles(out)
    assert any(f.severity == "critical" for f in out.findings)


def test_schema_agent_notices_a_timestamp_stored_as_text():
    t = make_tables()
    t["events"]["timestamp"] = t["events"]["timestamp"].astype(str)
    assert "events: timestamp is not a timestamp" in titles(SchemaAgent(ctx(t)).run(REQ))


def test_quality_agent_finds_duplicates_missing_constants_negatives_and_future_events():
    t = make_tables()
    t["events"].loc[5, "event_id"] = t["events"].loc[4, "event_id"]
    t["events"].loc[:600, "calls"] = np.nan
    t["events"]["flat"] = 1
    t["customers"].loc[0, "tenure_days"] = -5
    t["events"].loc[7, "timestamp"] = pd.Timestamp("2027-01-01")
    found = titles(QualityAgent(ctx(t)).run(REQ))
    assert "events: duplicate ids" in found
    assert any("events.calls" in x and "missing" in x for x in found)
    assert "events: constant columns" in found
    assert "customers.tenure_days: negative values" in found
    assert "events: timestamps outside the data window" in found


def test_relationship_agent_finds_orphans_hot_keys_duplicate_labels_and_short_history():
    t = make_tables()
    t["events"].loc[:200, "subscriber_id"] = "GHOST"  # 200 events from nobody, and one hot key
    t["labels"] = pd.concat([t["labels"], t["labels"].iloc[:3]])
    keep = (t["events"]["timestamp"] > pd.Timestamp("2026-03-01")) & t["events"]["subscriber_id"].isin([f"S{i}" for i in range(20)] + ["GHOST"])
    t["events"] = t["events"][keep]  # no history before the snapshots, and few customers covered
    found = titles(RelationshipAgent(ctx(t)).run(REQ))
    assert "events: rows without a customer" in found
    assert "Duplicate labels" in found
    assert "events: not enough history before the first snapshot" in found
    assert "events: low coverage" in found


def test_benjamini_hochberg_matches_a_hand_worked_example():
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.005]))
    assert adjusted == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert (benjamini_hochberg(np.array([0.5, 0.9])) <= 1).all()


def features_with_signal(seed=3, n=4000):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    frame = pd.DataFrame({f"noise__f{i}__7d": rng.normal(size=n) for i in range(30)})
    for i in range(6):
        frame[f"usage__s{i}__30d"] = rng.normal(size=n) + 0.9 * y
    frame["cat__plan"] = rng.choice(["a", "b"], n)
    return frame, y


def test_feature_agent_finds_the_real_group_and_ignores_noise_after_correction():
    frame, y = features_with_signal()
    out = FeatureAgent(ctx(features=frame, y=y, categorical=["cat__plan"])).run(REQ)
    groups = {g["group"]: g for g in out.summary["groups"]}
    assert groups["usage"]["survive"] == 6
    assert groups["noise"]["survive"] <= 1  # 30 noise features: correction keeps false hits near zero
    assert out.summary["groups"][0]["group"] == "usage"
    assert any(f.hypothesis and "usage" in f.hypothesis for f in out.findings)
    assert {w["window"] for w in out.summary["windows"]} == {"7d", "30d"}


def test_feature_agent_proposes_dropping_a_group_with_no_signal():
    frame, y = features_with_signal()
    out = FeatureAgent(ctx(features=frame, y=y, categorical=["cat__plan"])).run(REQ)
    assert any(f.title.startswith("No individual signal in noise") or f.title.startswith("Strong") for f in out.findings)


def test_feature_agent_flags_a_leaking_feature():
    frame, y = features_with_signal()
    frame["usage__leak__7d"] = y + np.random.default_rng(1).normal(scale=0.01, size=len(y))
    out = FeatureAgent(ctx(features=frame, y=y, categorical=["cat__plan"])).run(REQ)
    assert any(f.severity == "critical" and "usage__leak__7d" in f.title for f in out.findings)


def test_feature_agent_needs_labels_and_says_so():
    out = FeatureAgent(ctx()).run(REQ)
    assert out.findings[0].title == "No labelled features to test"


def test_swarm_runs_all_four_and_survives_a_crashing_agent():
    frame, y = features_with_signal()
    swarm = DataSwarm(ctx(features=frame, y=y, categorical=["cat__plan"]))
    swarm.agents[0].execute = lambda _r: (_ for _ in ()).throw(RuntimeError("boom"))
    report = swarm.run(REQ)
    assert "data_schema" in report.failed and {"data_quality", "data_relationships", "data_features"} <= set(report.summaries)
    assert report.markdown().startswith("# Data swarm report")


def test_telecom_specs_cover_all_six_raw_tables():
    specs = telecom_specs()
    assert set(specs) == {"customers", "usage_events", "recharges", "network_events", "risk_events", "snapshot_labels"}
    assert specs["customers"].categories["region"] == {"north", "south", "east", "west", "central"}


def test_a_statistically_real_but_tiny_effect_is_not_called_signal():
    rng = np.random.default_rng(9)
    n = 60_000
    y = (rng.random(n) < 0.1).astype(int)
    frame = pd.DataFrame({"tiny__f1__7d": rng.normal(size=n) + 0.03 * y, "big__f2__7d": rng.normal(size=n) + 0.8 * y})
    out = FeatureAgent(ctx(features=frame, y=y)).run(DataRequest(sample_rows=10_000, min_effect=0.03))
    groups = {g["group"]: g for g in out.summary["groups"]}
    assert groups["big"]["survive"] == 1 and groups["tiny"]["survive"] == 0


def test_activation_dates_are_not_checked_against_the_event_window():
    specs = telecom_specs()
    assert specs["customers"].event_time is False and specs["usage_events"].event_time is True
