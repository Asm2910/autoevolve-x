from __future__ import annotations

import pandas as pd


def build_recharge_features(
    recharges: pd.DataFrame,
) -> pd.DataFrame:
    """Build subscriber-level features from recharge events."""

    df = recharges.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    grouped = df.groupby("subscriber_id")

    features = grouped.agg(
        total_recharges=("recharge_id", "count"),
        total_recharge_amount=("recharge_amount", "sum"),
        avg_recharge_amount=("recharge_amount", "mean"),
        median_recharge_amount=("recharge_amount", "median"),
        max_recharge_amount=("recharge_amount", "max"),
        min_recharge_amount=("recharge_amount", "min"),
        recharge_amount_std=("recharge_amount", "std"),
        successful_recharges=("successful", "sum"),
        recharge_success_rate=("successful", "mean"),
    ).reset_index()

    features["failed_recharges"] = (
        features["total_recharges"]
        - features["successful_recharges"]
    )

    features["avg_amount_per_successful_recharge"] = (
        features["total_recharge_amount"]
        / features["successful_recharges"].clip(lower=1)
    )

    channel_features = (
        pd.crosstab(
            df["subscriber_id"],
            df["payment_channel"],
            normalize="index",
        )
        .add_prefix("payment_channel_rate_")
        .reset_index()
    )

    type_features = (
        pd.crosstab(
            df["subscriber_id"],
            df["recharge_type"],
            normalize="index",
        )
        .add_prefix("recharge_type_rate_")
        .reset_index()
    )

    features = features.merge(
        channel_features,
        on="subscriber_id",
        how="left",
    )

    features = features.merge(
        type_features,
        on="subscriber_id",
        how="left",
    )

    return features