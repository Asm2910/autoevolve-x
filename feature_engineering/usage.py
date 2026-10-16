from __future__ import annotations

import pandas as pd


def build_usage_features(
    usage_events: pd.DataFrame,
) -> pd.DataFrame:
    """Build subscriber-level features from usage events."""

    df = usage_events.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    grouped = df.groupby("subscriber_id")

    features = grouped.agg(
        total_usage_events=("event_id", "count"),
        total_calls=("call_count", "sum"),
        total_call_duration_sec=("call_duration_sec", "sum"),
        total_sms=("sms_count", "sum"),
        total_data_mb=("data_mb", "sum"),
        total_international_calls=("international_calls", "sum"),
        total_short_calls=("short_calls", "sum"),
        avg_call_duration_sec=("call_duration_sec", "mean"),
        avg_calls_per_event=("call_count", "mean"),
        avg_sms_per_event=("sms_count", "mean"),
        avg_data_mb_per_event=("data_mb", "mean"),
        avg_unique_destinations=("unique_destinations", "mean"),
        max_unique_destinations=("unique_destinations", "max"),
        night_activity_rate=("night_activity", "mean"),
        roaming_rate=("roaming_flag", "mean"),
    ).reset_index()

    features["international_call_ratio"] = (
        features["total_international_calls"]
        / features["total_calls"].clip(lower=1)
    )

    features["short_call_ratio"] = (
        features["total_short_calls"]
        / features["total_calls"].clip(lower=1)
    )

    features["avg_call_duration_per_call"] = (
        features["total_call_duration_sec"]
        / features["total_calls"].clip(lower=1)
    )

    return features