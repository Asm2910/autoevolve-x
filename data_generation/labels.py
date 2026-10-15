from __future__ import annotations

import numpy as np
import pandas as pd


def generate_labels(
    customers: pd.DataFrame,
    usage_events: pd.DataFrame,
    recharges: pd.DataFrame,
    network_events: pd.DataFrame,
    risk_events: pd.DataFrame,
    positive_rate: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate subscriber-level synthetic risk labels."""

    if not 0 < positive_rate < 1:
        raise ValueError("positive_rate must be between 0 and 1.")

    rng = np.random.default_rng(seed)

    subscriber_ids = customers["subscriber_id"].drop_duplicates()

    # ------------------------------------------------------------------
    # 1. Usage behavior
    # ------------------------------------------------------------------

    usage = (
        usage_events.groupby("subscriber_id")
        .agg(
            total_calls=("call_count", "sum"),
            total_call_duration=("call_duration_sec", "sum"),
            total_sms=("sms_count", "sum"),
            total_data_mb=("data_mb", "sum"),
            international_calls=("international_calls", "sum"),
            short_calls=("short_calls", "sum"),
            unique_destinations=("unique_destinations", "sum"),
            night_activity_rate=("night_activity", "mean"),
            roaming_rate=("roaming_flag", "mean"),
        )
        .reset_index()
    )

    usage["international_call_rate"] = (
        usage["international_calls"]
        / usage["total_calls"].clip(lower=1)
    )

    usage["short_call_rate"] = (
        usage["short_calls"]
        / usage["total_calls"].clip(lower=1)
    )

    # ------------------------------------------------------------------
    # 2. Recharge behavior
    # ------------------------------------------------------------------

    recharge = (
        recharges.groupby("subscriber_id")
        .agg(
            recharge_count=("recharge_id", "count"),
            recharge_amount=("recharge_amount", "sum"),
            recharge_failure_rate=(
                "successful",
                lambda x: 1.0 - x.mean(),
            ),
        )
        .reset_index()
    )

    recharge["avg_recharge_amount"] = (
        recharge["recharge_amount"]
        / recharge["recharge_count"].clip(lower=1)
    )

    # ------------------------------------------------------------------
    # 3. Network behavior
    # ------------------------------------------------------------------

    network = (
        network_events.groupby("subscriber_id")
        .agg(
            network_event_count=(
                "network_event_id",
                "count",
            ),
            avg_latency_ms=("latency_ms", "mean"),
            avg_packet_loss_pct=(
                "packet_loss_pct",
                "mean",
            ),
            avg_signal_strength=(
                "signal_strength",
                "mean",
            ),
            network_failure_rate=(
                "failed_connection",
                "mean",
            ),
            roaming_event_rate=(
                "roaming_flag",
                "mean",
            ),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 4. Historical risk behavior
    # ------------------------------------------------------------------

    risk = (
        risk_events.groupby("subscriber_id")
        .agg(
            risk_event_count=("risk_event_id", "count"),
            avg_risk_score=("risk_score", "mean"),
            max_risk_score=("risk_score", "max"),
            critical_risk_events=(
                "risk_level",
                lambda x: (x == "critical").sum(),
            ),
            high_risk_events=(
                "risk_level",
                lambda x: (x == "high").sum(),
            ),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 5. Combine all subscriber-level signals
    # ------------------------------------------------------------------

    features = pd.DataFrame(
        {"subscriber_id": subscriber_ids}
    )

    for dataframe in (
        usage,
        recharge,
        network,
        risk,
    ):
        features = features.merge(
            dataframe,
            on="subscriber_id",
            how="left",
        )

    numeric_columns = features.select_dtypes(
        include=np.number
    ).columns

    features[numeric_columns] = features[
        numeric_columns
    ].fillna(0)

    # ------------------------------------------------------------------
    # 6. Robust percentile-based anomaly signals
    # ------------------------------------------------------------------
    # Percentiles make the synthetic label less dependent on arbitrary
    # absolute thresholds.

    def percentile_score(column: str) -> pd.Series:
        values = features[column]

        return values.rank(
            pct=True,
            method="average",
        )

    features["usage_risk"] = (
        0.30 * percentile_score("total_calls")
        + 0.20 * percentile_score(
            "international_call_rate"
        )
        + 0.20 * percentile_score(
            "short_call_rate"
        )
        + 0.15 * percentile_score(
            "unique_destinations"
        )
        + 0.15 * percentile_score(
            "night_activity_rate"
        )
    )

    features["recharge_risk"] = (
        0.50 * percentile_score(
            "recharge_failure_rate"
        )
        + 0.30 * percentile_score(
            "recharge_count"
        )
        + 0.20 * percentile_score(
            "avg_recharge_amount"
        )
    )

    features["network_risk"] = (
        0.35 * percentile_score(
            "network_failure_rate"
        )
        + 0.25 * percentile_score(
            "avg_latency_ms"
        )
        + 0.20 * percentile_score(
            "avg_packet_loss_pct"
        )
        + 0.20
        * (
            1
            - percentile_score(
                "avg_signal_strength"
            )
        )
    )

    features["historical_risk"] = (
        0.45 * percentile_score(
            "avg_risk_score"
        )
        + 0.25 * percentile_score(
            "max_risk_score"
        )
        + 0.20 * percentile_score(
            "risk_event_count"
        )
        + 0.10 * percentile_score(
            "critical_risk_events"
        )
    )

    # ------------------------------------------------------------------
    # 7. Latent risk score
    # ------------------------------------------------------------------

    latent_risk = (
        0.35 * features["usage_risk"]
        + 0.15 * features["recharge_risk"]
        + 0.20 * features["network_risk"]
        + 0.30 * features["historical_risk"]
    )

    # Small noise prevents a perfectly deterministic target.
    noise = rng.normal(
        loc=0.0,
        scale=0.03,
        size=len(features),
    )

    features["latent_risk_score"] = np.clip(
        latent_risk + noise,
        0,
        1,
    )

    # ------------------------------------------------------------------
    # 8. Generate controlled class imbalance
    # ------------------------------------------------------------------

    threshold = features[
        "latent_risk_score"
    ].quantile(1 - positive_rate)

    features["risk_label"] = (
        features["latent_risk_score"] >= threshold
    ).astype(int)

    # ------------------------------------------------------------------
    # 9. Final label dataset
    # ------------------------------------------------------------------

    labels = features[
        [
            "subscriber_id",
            "risk_label",
        ]
    ].copy()

    return labels