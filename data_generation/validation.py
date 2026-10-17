"""Validation for the raw v2 datasets.

Every check appends to an error list, so one run reports every problem.
"""

from __future__ import annotations

import pandas as pd

from data_generation.snapshots import snapshot_dates

EXPECTED_COLUMNS = {
    "customers": [
        "subscriber_id", "activation_date", "customer_segment", "plan_type", "region",
        "device_type", "tenure_days", "age_band", "preferred_language", "account_status",
    ],
    "usage_events": [
        "event_id", "subscriber_id", "timestamp", "call_count", "call_duration_sec",
        "sms_count", "data_mb", "international_calls", "short_calls",
        "unique_destinations", "night_activity", "roaming_flag",
    ],
    "recharges": [
        "recharge_id", "subscriber_id", "timestamp", "recharge_amount",
        "payment_channel", "recharge_type", "successful",
    ],
    "network_events": [
        "network_event_id", "subscriber_id", "timestamp", "cell_id", "network_type",
        "signal_strength", "latency_ms", "packet_loss_pct", "data_session_minutes",
        "roaming_flag", "failed_connection",
    ],
    "risk_events": [
        "risk_event_id", "subscriber_id", "timestamp", "risk_type", "risk_score",
        "risk_level", "trigger_count", "source_system",
    ],
    "snapshot_labels": ["subscriber_id", "snapshot_date", "risk_label"],
}

ID_COLUMN = {
    "customers": "subscriber_id",
    "usage_events": "event_id",
    "recharges": "recharge_id",
    "network_events": "network_event_id",
    "risk_events": "risk_event_id",
}

CONFIG_KEY = {
    "customers": "customers",
    "usage_events": "usage_events",
    "recharges": "recharges",
    "network_events": "network_events",
    "risk_events": "risk_events",
}

CATEGORIES = {
    ("customers", "customer_segment"): {"consumer", "business", "premium"},
    ("customers", "plan_type"): {"prepaid", "postpaid"},
    ("customers", "region"): {"north", "south", "east", "west", "central"},
    ("customers", "device_type"): {"smartphone", "feature_phone"},
    ("customers", "age_band"): {"18_25", "26_35", "36_45", "46_60", "60_plus"},
    ("customers", "preferred_language"): {"english", "french", "arabic", "local"},
    ("customers", "account_status"): {"active", "suspended", "inactive"},
    ("recharges", "payment_channel"): {"mobile_app", "ussd", "retail_store", "bank", "web"},
    ("recharges", "recharge_type"): {
        "standard", "bundle", "emergency", "bill_payment", "advance_payment",
    },
    ("network_events", "network_type"): {"3G", "4G", "5G"},
    ("risk_events", "risk_type"): {
        "usage_anomaly", "network_anomaly", "payment_anomaly",
        "account_anomaly", "behavioral_anomaly", "location_anomaly",
    },
    ("risk_events", "risk_level"): {"low", "medium", "high", "critical"},
    ("risk_events", "source_system"): {
        "fraud_engine", "network_monitor", "billing_system", "behavior_engine",
    },
}

PREPAID_TYPES = {"standard", "bundle", "emergency"}
POSTPAID_TYPES = {"bill_payment", "advance_payment"}


def _check(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _check_structure(datasets, config, errors) -> bool:
    """Return False when columns are wrong (later checks would just crash)."""
    columns_ok = True
    for name, columns in EXPECTED_COLUMNS.items():
        frame = datasets[name]
        if list(frame.columns) != columns:
            errors.append(f"{name}: unexpected columns {list(frame.columns)}")
            columns_ok = False
        _check(errors, not frame.isna().any().any(), f"{name}: contains missing values")
    if not columns_ok:
        return False

    for name, id_column in ID_COLUMN.items():
        _check(errors, datasets[name][id_column].is_unique, f"{name}: {id_column} is not unique")
        expected_rows = config["data"][CONFIG_KEY[name]]["n_rows"]
        _check(errors, len(datasets[name]) == expected_rows, f"{name}: expected {expected_rows} rows, found {len(datasets[name])}")

    for (name, column), allowed in CATEGORIES.items():
        found = set(datasets[name][column].unique())
        _check(errors, found <= allowed, f"{name}.{column}: unexpected values {sorted(found - allowed)}")
    return True


def _check_references(datasets, errors) -> None:
    known = set(datasets["customers"]["subscriber_id"])
    for name in ("usage_events", "recharges", "network_events", "risk_events", "snapshot_labels"):
        unknown = set(datasets[name]["subscriber_id"]) - known
        _check(errors, not unknown, f"{name}: {len(unknown)} unknown subscribers")


def _check_time(datasets, config, errors) -> None:
    start = pd.Timestamp(config["data"]["time_range"]["start"])
    end = pd.Timestamp(config["data"]["time_range"]["end"])
    activation = datasets["customers"].set_index("subscriber_id")["activation_date"]

    for name in ("usage_events", "recharges", "network_events", "risk_events"):
        frame = datasets[name]
        _check(errors, frame["timestamp"].min() >= start, f"{name}: events before window start")
        _check(errors, frame["timestamp"].max() <= end, f"{name}: events after window end")
        before_activation = (frame["timestamp"] < frame["subscriber_id"].map(activation)).sum()
        _check(errors, before_activation == 0, f"{name}: {before_activation} events before activation")
        _check(errors, frame["timestamp"].is_monotonic_increasing, f"{name}: not sorted by timestamp")

    customers = datasets["customers"]
    expected_tenure = (end - customers["activation_date"]).dt.days
    _check(errors, (customers["tenure_days"] == expected_tenure).all(), "customers: tenure_days inconsistent with activation_date")


def _check_values(datasets, errors) -> None:
    usage = datasets["usage_events"]
    for column in ("call_count", "call_duration_sec", "sms_count", "data_mb"):
        _check(errors, (usage[column] >= 0).all(), f"usage_events.{column}: negative values")
    _check(errors, (usage["international_calls"] <= usage["call_count"]).all(), "usage_events: international_calls exceed call_count")
    _check(errors, (usage["short_calls"] <= usage["call_count"]).all(), "usage_events: short_calls exceed call_count")
    _check(errors, (usage["unique_destinations"] >= 1).all(), "usage_events: unique_destinations below 1")
    for column in ("night_activity", "roaming_flag"):
        _check(errors, usage[column].isin([0, 1]).all(), f"usage_events.{column}: must be 0/1")

    recharges = datasets["recharges"]
    _check(errors, (recharges["recharge_amount"] > 0).all(), "recharges: non-positive amounts")
    _check(errors, recharges["successful"].isin([0, 1]).all(), "recharges.successful must be 0/1")
    plan = recharges["subscriber_id"].map(datasets["customers"].set_index("subscriber_id")["plan_type"])
    prepaid_ok = recharges.loc[plan == "prepaid", "recharge_type"].isin(PREPAID_TYPES).all()
    postpaid_ok = recharges.loc[plan == "postpaid", "recharge_type"].isin(POSTPAID_TYPES).all()
    _check(errors, prepaid_ok and postpaid_ok, "recharges: recharge_type inconsistent with plan_type")

    network = datasets["network_events"]
    _check(errors, network["signal_strength"].between(-120, -40).all(), "network_events: signal_strength out of range")
    _check(errors, (network["latency_ms"] > 0).all(), "network_events: non-positive latency")
    _check(errors, (network["packet_loss_pct"] >= 0).all(), "network_events: negative packet loss")
    for column in ("roaming_flag", "failed_connection"):
        _check(errors, network[column].isin([0, 1]).all(), f"network_events.{column}: must be 0/1")

    risk = datasets["risk_events"]
    _check(errors, risk["risk_score"].between(0, 100).all(), "risk_events: risk_score out of range")
    _check(errors, (risk["trigger_count"] >= 1).all(), "risk_events: trigger_count below 1")
    expected_level = pd.cut(
        risk["risk_score"], [-0.01, 34.999, 59.999, 79.999, 100.0],
        labels=["low", "medium", "high", "critical"],
    ).astype(str)
    _check(errors, (expected_level == risk["risk_level"]).mean() > 0.999, "risk_events: risk_level inconsistent with risk_score")


def _check_labels(datasets, config, errors) -> None:
    labels = datasets["snapshot_labels"]
    _check(errors, labels["risk_label"].isin([0, 1]).all(), "snapshot_labels: risk_label must be 0/1")
    _check(errors, not labels.duplicated(["subscriber_id", "snapshot_date"]).any(), "snapshot_labels: duplicate (subscriber, snapshot) rows")

    expected = set(snapshot_dates(config))
    found = set(labels["snapshot_date"].unique())
    _check(errors, found == expected, "snapshot_labels: snapshot dates do not match config")

    rate = labels["risk_label"].mean()
    _check(errors, 0.002 < rate < 0.03, f"snapshot_labels: implausible positive rate {rate:.4f}")
    per_snapshot = labels.groupby("snapshot_date")["risk_label"].sum()
    _check(errors, (per_snapshot >= 20).all(), "snapshot_labels: a snapshot has fewer than 20 positives")

    activation = datasets["customers"].set_index("subscriber_id")["activation_date"]
    activated_late = (labels["snapshot_date"] < labels["subscriber_id"].map(activation)).sum()
    _check(errors, activated_late == 0, f"snapshot_labels: {activated_late} rows before activation")


def validate_raw_datasets(datasets: dict[str, pd.DataFrame], config: dict) -> list[str]:
    """Return a list of problems; an empty list means the data is valid."""
    errors: list[str] = []
    if not _check_structure(datasets, config, errors):
        return errors
    _check_references(datasets, errors)
    _check_time(datasets, config, errors)
    _check_values(datasets, errors)
    _check_labels(datasets, config, errors)
    return errors
