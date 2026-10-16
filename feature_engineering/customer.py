import pandas as pd


def build_customer_features(customers: pd.DataFrame) -> pd.DataFrame:
    """Build subscriber-level features from customer attributes."""

    required_columns = {
        "subscriber_id",
        "activation_date",
        "customer_segment",
        "plan_type",
        "region",
        "device_type",
        "tenure_days",
        "age_band",
        "preferred_language",
        "account_status",
    }

    missing_columns = required_columns - set(customers.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    features = customers[
        [
            "subscriber_id",
            "customer_segment",
            "plan_type",
            "region",
            "device_type",
            "tenure_days",
            "age_band",
            "preferred_language",
            "account_status",
        ]
    ].copy()

    features["is_prepaid"] = (
        features["plan_type"] == "prepaid"
    ).astype(int)

    features["is_postpaid"] = (
        features["plan_type"] == "postpaid"
    ).astype(int)

    features["is_business"] = (
        features["customer_segment"] == "business"
    ).astype(int)

    features["is_premium"] = (
        features["customer_segment"] == "premium"
    ).astype(int)

    features["is_smartphone"] = (
        features["device_type"] == "smartphone"
    ).astype(int)

    features["is_active"] = (
        features["account_status"] == "active"
    ).astype(int)

    return features