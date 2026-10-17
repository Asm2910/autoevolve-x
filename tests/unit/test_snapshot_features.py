"""Unit tests proving snapshot features only use the past."""

import pandas as pd

from feature_engineering.builder import build_feature_dataset

AS_OF = pd.Timestamp("2025-09-30")


def days_before(days: float) -> pd.Timestamp:
    return AS_OF - pd.Timedelta(days=days)


def make_customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subscriber_id": ["A", "B"],
            "activation_date": pd.to_datetime(["2025-01-01", "2025-06-01"]),
            "customer_segment": ["consumer", "business"],
            "plan_type": ["prepaid", "postpaid"],
            "region": ["north", "south"],
            "device_type": ["smartphone", "feature_phone"],
            "tenure_days": [999, 999],  # stale on purpose: must be recomputed as-of
            "age_band": ["26_35", "36_45"],
            "preferred_language": ["english", "french"],
            "account_status": ["active", "active"],
        }
    )


def usage_events(timestamps) -> pd.DataFrame:
    n = len(timestamps)
    return pd.DataFrame(
        {
            "event_id": [f"E{i}" for i in range(n)],
            "subscriber_id": ["A"] * n,
            "timestamp": pd.to_datetime(timestamps),
            "call_count": [5] * n,
            "call_duration_sec": [100.0] * n,
            "sms_count": [1] * n,
            "data_mb": [10.0] * n,
            "international_calls": [1] * n,
            "short_calls": [2] * n,
            "unique_destinations": [3] * n,
            "night_activity": [0] * n,
            "roaming_flag": [0] * n,
        }
    )


def recharges(timestamps) -> pd.DataFrame:
    n = len(timestamps)
    return pd.DataFrame(
        {
            "recharge_id": [f"R{i}" for i in range(n)],
            "subscriber_id": ["A"] * n,
            "timestamp": pd.to_datetime(timestamps),
            "recharge_amount": [10.0] * n,
            "payment_channel": ["ussd"] * n,
            "recharge_type": ["standard"] * n,
            "successful": [1] * n,
        }
    )


def network_events(timestamps) -> pd.DataFrame:
    n = len(timestamps)
    return pd.DataFrame(
        {
            "network_event_id": [f"N{i}" for i in range(n)],
            "subscriber_id": ["A"] * n,
            "timestamp": pd.to_datetime(timestamps),
            "cell_id": ["CELL_00001"] * n,
            "network_type": ["4G"] * n,
            "signal_strength": [-70.0] * n,
            "latency_ms": [30.0] * n,
            "packet_loss_pct": [0.5] * n,
            "data_session_minutes": [5.0] * n,
            "roaming_flag": [0] * n,
            "failed_connection": [0] * n,
        }
    )


def risk_events(timestamps) -> pd.DataFrame:
    n = len(timestamps)
    return pd.DataFrame(
        {
            "risk_event_id": [f"K{i}" for i in range(n)],
            "subscriber_id": ["A"] * n,
            "timestamp": pd.to_datetime(timestamps),
            "risk_type": ["usage_anomaly"] * n,
            "risk_score": [50.0] * n,
            "risk_level": ["medium"] * n,
            "trigger_count": [2] * n,
            "source_system": ["fraud_engine"] * n,
        }
    )


def build(usage_times, other_times=None):
    other_times = other_times if other_times is not None else [days_before(3)]
    labels = pd.DataFrame(
        {"subscriber_id": ["A", "B"], "snapshot_date": [AS_OF, AS_OF], "risk_label": [0, 1]}
    )
    return build_feature_dataset(
        customers=make_customers(),
        usage_events=usage_events(usage_times),
        recharges=recharges(other_times),
        network_events=network_events(other_times),
        risk_events=risk_events(other_times),
        snapshot_labels=labels,
    ).set_index("subscriber_id")


def test_each_window_counts_only_its_own_events():
    times = [days_before(3), days_before(20), days_before(60), days_before(100)]
    row = build(times).loc["A"]

    assert row["usage__total_usage_events__7d"] == 1
    assert row["usage__total_usage_events__30d"] == 2
    assert row["usage__total_usage_events__90d"] == 3  # the 100-day-old event is outside


def test_future_events_do_not_change_features():
    past = [days_before(3), days_before(20)]
    future = [AS_OF + pd.Timedelta(days=1), AS_OF + pd.Timedelta(days=15)]

    without_future = build(past)
    with_future = build(past + future, other_times=[days_before(3), AS_OF + pd.Timedelta(days=2)])
    expected = build(past, other_times=[days_before(3)])

    pd.testing.assert_frame_equal(without_future, expected)
    pd.testing.assert_frame_equal(with_future, expected)


def test_tenure_is_computed_as_of_the_snapshot():
    features = build([days_before(3)])

    assert features.loc["A", "customer__tenure_days"] == (AS_OF - pd.Timestamp("2025-01-01")).days
    assert features.loc["B", "customer__tenure_days"] == (AS_OF - pd.Timestamp("2025-06-01")).days


def test_subscriber_without_events_gets_zero_counts_and_missing_averages():
    row = build([days_before(3)]).loc["B"]

    assert row["usage__has_events__30d"] == 0
    assert row["usage__total_calls__30d"] == 0
    assert pd.isna(row["usage__avg_call_duration_sec__30d"])


def test_label_and_snapshot_date_are_kept():
    features = build([days_before(3)])

    assert features.loc["B", "risk_label"] == 1
    assert (features["snapshot_date"] == AS_OF).all()
