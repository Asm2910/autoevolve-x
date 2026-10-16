import pandas as pd

from feature_engineering.customer import build_customer_features
from feature_engineering.network import build_network_features
from feature_engineering.recharges import build_recharge_features
from feature_engineering.risk import build_risk_features
from feature_engineering.usage import build_usage_features


def build_feature_dataset(
    customers: pd.DataFrame,
    usage_events: pd.DataFrame,
    recharges: pd.DataFrame,
    network_events: pd.DataFrame,
    risk_events: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Build the final subscriber-level modeling dataset."""

    customer_features = build_customer_features(customers)
    usage_features = build_usage_features(usage_events)
    recharge_features = build_recharge_features(recharges)
    network_features = build_network_features(network_events)
    risk_features = build_risk_features(risk_events)

    feature_groups = [
        usage_features,
        recharge_features,
        network_features,
        risk_features,
    ]

    final_features = customer_features.copy()

    for features in feature_groups:
        final_features = final_features.merge(
            features,
            on="subscriber_id",
            how="left",
            validate="one_to_one",
        )

    final_features = final_features.merge(
        labels,
        on="subscriber_id",
        how="left",
        validate="one_to_one",
    )

    return final_features


def fill_missing_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Fill missing event-derived features for subscribers without events."""

    result = features.copy()

    event_prefixes = (
        "total_",
        "avg_",
        "max_",
        "min_",
        "median_",
        "successful_",
        "failed_",
        "recharge_",
        "payment_channel_",
        "recharge_type_",
        "roaming_",
        "connection_",
        "poor_signal",
        "network_type_",
        "risk_",
        "high_risk_",
        "critical_risk_",
        "international_call_",
        "short_call_",
        "night_activity_",
    )

    numeric_columns = result.select_dtypes(
        include=["number"]
    ).columns

    for column in numeric_columns:
        if column == "risk_label":
            continue

        if column.startswith(event_prefixes):
            result[column] = result[column].fillna(0)

    return result