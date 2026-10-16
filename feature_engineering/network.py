from __future__ import annotations

import pandas as pd


def build_network_features(
    network_events: pd.DataFrame,
) -> pd.DataFrame:
    """Build subscriber-level features from network events."""

    df = network_events.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    grouped = df.groupby("subscriber_id")

    features = grouped.agg(
        total_network_events=("network_event_id", "count"),
        avg_signal_strength=("signal_strength", "mean"),
        min_signal_strength=("signal_strength", "min"),
        max_signal_strength=("signal_strength", "max"),
        avg_latency_ms=("latency_ms", "mean"),
        max_latency_ms=("latency_ms", "max"),
        avg_packet_loss_pct=("packet_loss_pct", "mean"),
        max_packet_loss_pct=("packet_loss_pct", "max"),
        total_session_minutes=("data_session_minutes", "sum"),
        avg_session_minutes=("data_session_minutes", "mean"),
        roaming_rate=("roaming_flag", "mean"),
        connection_failure_rate=("failed_connection", "mean"),
    ).reset_index()

    features["poor_signal_rate"] = (
        df.assign(
            poor_signal=df["signal_strength"] < -95,
        )
        .groupby("subscriber_id")["poor_signal"]
        .mean()
        .reset_index(drop=True)
    )

    network_type_rates = (
        pd.crosstab(
            df["subscriber_id"],
            df["network_type"],
            normalize="index",
        )
        .add_prefix("network_type_rate_")
        .reset_index()
    )

    features = features.merge(
        network_type_rates,
        on="subscriber_id",
        how="left",
    )

    return features