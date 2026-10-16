import pandas as pd


def build_risk_features(risk_events: pd.DataFrame) -> pd.DataFrame:
    """Build subscriber-level features from risk events."""

    required_columns = {
        "subscriber_id",
        "risk_score",
        "risk_level",
        "risk_type",
        "trigger_count",
    }

    missing_columns = required_columns - set(risk_events.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    risk = risk_events.copy()

    risk["is_high_risk"] = (
        risk["risk_level"].isin(["high", "critical"]).astype(int)
    )

    risk["is_critical_risk"] = (
        risk["risk_level"] == "critical"
    ).astype(int)

    features = (
        risk.groupby("subscriber_id")
        .agg(
            total_risk_events=("risk_score", "size"),
            avg_risk_score=("risk_score", "mean"),
            max_risk_score=("risk_score", "max"),
            min_risk_score=("risk_score", "min"),
            risk_score_std=("risk_score", "std"),
            avg_trigger_count=("trigger_count", "mean"),
            max_trigger_count=("trigger_count", "max"),
            high_risk_event_count=("is_high_risk", "sum"),
            critical_risk_event_count=("is_critical_risk", "sum"),
        )
        .reset_index()
    )

    features["risk_score_std"] = features["risk_score_std"].fillna(0)

    features["high_risk_event_rate"] = (
        features["high_risk_event_count"]
        / features["total_risk_events"]
    )

    features["critical_risk_event_rate"] = (
        features["critical_risk_event_count"]
        / features["total_risk_events"]
    )

    risk_type_rates = pd.crosstab(
        risk["subscriber_id"],
        risk["risk_type"],
        normalize="index",
    )

    risk_type_rates.columns = [
        f"risk_type_rate_{column}"
        for column in risk_type_rates.columns
    ]

    risk_type_rates = risk_type_rates.reset_index()

    features = features.merge(
        risk_type_rates,
        on="subscriber_id",
        how="left",
    )

    return features