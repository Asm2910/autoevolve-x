from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_generation.profiles import generate_behavior_profiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

def load_config() -> dict:
    """Load project data-generation configuration."""
    with open(CONFIG_PATH, encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_customers(config: dict) -> pd.DataFrame:
    """Generate synthetic customer master data."""
    seed = config["data"]["seed"]
    n_rows = config["data"]["customers"]["n_rows"]

    rng = np.random.default_rng(seed)

    subscriber_ids = [
        f"SUB{i:06d}"
        for i in range(1, n_rows + 1)
    ]

    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2026-01-01")

    activation_seconds = int(
        (end_date - start_date).total_seconds()
    )

    activation_dates = start_date + pd.to_timedelta(
        rng.integers(
            0,
            activation_seconds,
            size=n_rows,
        ),
        unit="s",
    )

    customer_segment = rng.choice(
        [
            "consumer",
            "business",
            "premium",
        ],
        size=n_rows,
        p=[
            0.65,
            0.25,
            0.10,
        ],
    )

    plan_type = rng.choice(
        [
            "prepaid",
            "postpaid",
        ],
        size=n_rows,
        p=[
            0.70,
            0.30,
        ],
    )

    region = rng.choice(
        [
            "north",
            "south",
            "east",
            "west",
            "central",
        ],
        size=n_rows,
        p=[
            0.20,
            0.20,
            0.15,
            0.25,
            0.20,
        ],
    )

    device_type = rng.choice(
        [
            "smartphone",
            "feature_phone",
        ],
        size=n_rows,
        p=[
            0.75,
            0.25,
        ],
    )

    current_date = pd.Timestamp("2026-06-30")

    tenure_days = (
        current_date - activation_dates
    ).days

    age_band = rng.choice(
        [
            "18_25",
            "26_35",
            "36_45",
            "46_60",
            "60_plus",
        ],
        size=n_rows,
        p=[
            0.20,
            0.30,
            0.25,
            0.15,
            0.10,
        ],
    )

    preferred_language = rng.choice(
        [
            "english",
            "french",
            "arabic",
            "local",
        ],
        size=n_rows,
        p=[
            0.20,
            0.30,
            0.25,
            0.25,
        ],
    )

    account_status = rng.choice(
        [
            "active",
            "suspended",
            "inactive",
        ],
        size=n_rows,
        p=[
            0.90,
            0.05,
            0.05,
        ],
    )

    return pd.DataFrame(
        {
            "subscriber_id": subscriber_ids,
            "activation_date": activation_dates,
            "customer_segment": customer_segment,
            "plan_type": plan_type,
            "region": region,
            "device_type": device_type,
            "tenure_days": tenure_days,
            "age_band": age_band,
            "preferred_language": preferred_language,
            "account_status": account_status,
        }
    )


def generate_usage_events(
    config: dict,
    customers: pd.DataFrame,
    behavior_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Generate synthetic telecom usage events."""
    seed = config["data"]["seed"] + 1
    n_rows = config["data"]["usage_events"]["n_rows"]

    rng = np.random.default_rng(seed)

    subscriber_ids = customers["subscriber_id"].to_numpy()

    selected_subscribers = rng.choice(
        subscriber_ids,
        size=n_rows,
        replace=True,
    )

    profile_index = behavior_profiles.set_index(
        "subscriber_id"
    )

    selected_profiles = profile_index.loc[
        selected_subscribers
    ].reset_index()

    # usage_intensity = selected_profiles[
    #     "usage_propensity"
    # ].to_numpy()

    international_tendency = selected_profiles[
        "roaming_propensity"
    ].to_numpy()

    night_tendency = selected_profiles[
        "usage_propensity"
    ].to_numpy()

    destination_tendency = selected_profiles[
        "usage_propensity"
    ].to_numpy()

    start_date = pd.Timestamp(
        config["data"]["time_range"]["start"]
    )

    end_date = pd.Timestamp(
        config["data"]["time_range"]["end"]
    )

    total_seconds = int(
        (end_date - start_date).total_seconds()
    )

    timestamps = start_date + pd.to_timedelta(
        rng.integers(
            0,
            total_seconds,
            size=n_rows,
        ),
        unit="s",
    )

    intensity = selected_profiles[
        "usage_propensity"
    ].to_numpy()

    international_tendency = selected_profiles[
        "roaming_propensity"
    ].to_numpy()

    night_tendency = selected_profiles[
        "usage_propensity"
    ].to_numpy()

    destination_tendency = selected_profiles[
        "usage_propensity"
    ].to_numpy()

    call_count = rng.poisson(
        2 + 12 * intensity
    )

    call_duration_sec = (
        call_count
        * rng.gamma(
            shape=2.0,
            scale=45.0,
            size=n_rows,
        )
    )

    sms_count = rng.poisson(
        1 + 8 * intensity
    )

    data_mb = (
        rng.gamma(
            shape=2.0,
            scale=150.0,
            size=n_rows,
        )
        * (0.5 + intensity)
    )

    international_calls = rng.binomial(
        np.maximum(call_count, 1),
        np.clip(
            0.02 + 0.35 * international_tendency,
            0,
            1,
        ),
    )

    short_call_probability = np.clip(
        0.15
        + 0.35 * international_tendency
        + 0.15 * destination_tendency,
        0,
        0.95,
    )

    short_calls = rng.binomial(
        np.maximum(call_count, 1),
        short_call_probability,
    )

    unique_destinations = np.maximum(
        1,
        rng.poisson(
            1 + 8 * destination_tendency
        ),
    )

    night_activity = rng.binomial(
        1,
        np.clip(
            0.10 + 0.65 * night_tendency,
            0,
            0.95,
        ),
    )

    roaming_flag = rng.binomial(
        1,
        np.clip(
            0.05 + 0.10 * international_tendency,
            0,
            1,
        ),
    )

    event_ids = [
        f"EVT{i:08d}"
        for i in range(1, n_rows + 1)
    ]

    return pd.DataFrame(
        {
            "event_id": event_ids,
            "subscriber_id": selected_subscribers,
            "timestamp": timestamps,
            "call_count": call_count,
            "call_duration_sec": call_duration_sec,
            "sms_count": sms_count,
            "data_mb": data_mb,
            "international_calls": international_calls,
            "short_calls": short_calls,
            "unique_destinations": unique_destinations,
            "night_activity": night_activity,
            "roaming_flag": roaming_flag,
        }
    )


def generate_recharges(
    config: dict,
    customers: pd.DataFrame,
    behavior_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Generate synthetic recharge events."""
    seed = config["data"]["seed"] + 2
    n_rows = config["data"]["recharges"]["n_rows"]

    rng = np.random.default_rng(seed)

    subscriber_ids = customers["subscriber_id"].to_numpy()

    customer_index = customers.set_index(
        "subscriber_id"
    )

    customer_plan = customer_index["plan_type"]

    customer_segment = customer_index["customer_segment"]

    recharge_profile = behavior_profiles.set_index(
        "subscriber_id"
    )["recharge_propensity"]

    selected_subscribers = rng.choice(
        subscriber_ids,
        size=n_rows,
        replace=True,
    )

    selected_plans = customer_plan.loc[
        selected_subscribers
    ].to_numpy()

    selected_segments = customer_segment.loc[
        selected_subscribers
    ].to_numpy()

    selected_recharge_propensity = (
        recharge_profile.loc[
            selected_subscribers
        ].to_numpy()
    )

    start_date = pd.Timestamp(
        config["data"]["time_range"]["start"]
    )

    end_date = pd.Timestamp(
        config["data"]["time_range"]["end"]
    )

    total_seconds = int(
        (end_date - start_date).total_seconds()
    )

    timestamps = start_date + pd.to_timedelta(
        rng.integers(
            0,
            total_seconds,
            size=n_rows,
        ),
        unit="s",
    )

    prepaid_mask = selected_plans == "prepaid"
    business_mask = selected_segments == "business"
    premium_mask = selected_segments == "premium"

    recharge_amount = np.zeros(n_rows)

    prepaid_amounts = rng.lognormal(
        mean=np.log(12),
        sigma=0.55,
        size=n_rows,
    )

    postpaid_amounts = rng.lognormal(
        mean=np.log(45),
        sigma=0.45,
        size=n_rows,
    )

    recharge_amount[prepaid_mask] = (
        prepaid_amounts[prepaid_mask]
    )

    recharge_amount[~prepaid_mask] = (
        postpaid_amounts[~prepaid_mask]
    )

    recharge_amount *= np.where(
        business_mask,
        1.25,
        1.0,
    )

    recharge_amount *= np.where(
        premium_mask,
        1.50,
        1.0,
    )

    recharge_amount *= (
        0.70
        + 0.90 * selected_recharge_propensity
    )

    recharge_amount = np.round(
        recharge_amount,
        2,
    )

    channel_score = rng.random(n_rows)

    payment_channel = np.where(
        channel_score
        < 0.20 + 0.25 * selected_recharge_propensity,
        "mobile_app",
        np.where(
            channel_score
            < 0.45 + 0.15 * selected_recharge_propensity,
            "ussd",
            np.where(
                channel_score < 0.70,
                "retail_store",
                np.where(
                    channel_score < 0.90,
                    "bank",
                    "web",
                ),
            ),
        ),
    )

    prepaid_recharge_probability = np.clip(
        0.55
        + 0.25 * selected_recharge_propensity,
        0,
        1,
    )

    prepaid_type_random = rng.random(n_rows)

    recharge_type = np.where(
        prepaid_mask,
        np.where(
            prepaid_type_random
            < prepaid_recharge_probability,
            "standard",
            np.where(
                prepaid_type_random < 0.90,
                "bundle",
                "emergency",
            ),
        ),
        np.where(
            rng.random(n_rows) < 0.85,
            "bill_payment",
            "advance_payment",
        ),
    )

    failure_probability = np.where(
        payment_channel == "retail_store",
        0.04,
        0.02,
    )

    failure_probability *= (
        1.20
        - 0.40 * selected_recharge_propensity
    )

    failure_probability = np.clip(
        failure_probability,
        0.005,
        0.10,
    )

    successful = (
        rng.random(n_rows)
        > failure_probability
    ).astype(int)

    event_ids = [
        f"RCH{i:08d}"
        for i in range(1, n_rows + 1)
    ]

    return pd.DataFrame(
        {
            "recharge_id": event_ids,
            "subscriber_id": selected_subscribers,
            "timestamp": timestamps,
            "recharge_amount": recharge_amount,
            "payment_channel": payment_channel,
            "recharge_type": recharge_type,
            "successful": successful,
        }
    )


def generate_network_events(
    config: dict,
    customers: pd.DataFrame,
    behavior_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Generate synthetic network events."""
    seed = config["data"]["seed"] + 3
    n_rows = config["data"]["network_events"]["n_rows"]

    rng = np.random.default_rng(seed)

    subscriber_ids = customers["subscriber_id"].to_numpy()

    selected_subscribers = rng.choice(
        subscriber_ids,
        size=n_rows,
        replace=True,
    )

    network_profile = behavior_profiles.set_index(
        "subscriber_id"
    )["network_instability"]

    selected_instability = network_profile.loc[
        selected_subscribers
    ].to_numpy()

    start_date = pd.Timestamp(
        config["data"]["time_range"]["start"]
    )

    end_date = pd.Timestamp(
        config["data"]["time_range"]["end"]
    )

    total_seconds = int(
        (end_date - start_date).total_seconds()
    )

    timestamps = start_date + pd.to_timedelta(
        rng.integers(
            0,
            total_seconds,
            size=n_rows,
        ),
        unit="s",
    )

    cell_ids = [
        f"CELL_{i:05d}"
        for i in rng.integers(
            1,
            501,
            size=n_rows,
        )
    ]

    network_type = rng.choice(
        [
            "3G",
            "4G",
            "5G",
        ],
        size=n_rows,
        p=[
            0.10,
            0.55,
            0.35,
        ],
    )

    signal_strength = (
        rng.normal(
            -75,
            10,
            size=n_rows,
        )
        - 25 * selected_instability
    )

    signal_strength = np.clip(
        signal_strength,
        -120,
        -40,
    )

    latency_ms = (
        rng.gamma(
            shape=2.0,
            scale=15.0,
            size=n_rows,
        )
        * (
            1.0
            + 1.5 * selected_instability
        )
    )

    packet_loss_pct = (
        rng.beta(
            2.0,
            5.0,
            size=n_rows,
        )
        * 2.5
        * (
            1.0
            + 2.0 * selected_instability
        )
    )

    data_session_minutes = (
        rng.gamma(
            shape=2.0,
            scale=8.0,
            size=n_rows,
        )
        * (
            0.7
            + 0.8 * (1 - selected_instability)
        )
    )

    roaming_flag = rng.binomial(
        1,
        np.clip(
            0.04
            + 0.08 * selected_instability,
            0,
            1,
        ),
    )

    failed_connection_probability = np.clip(
        0.008
        + 0.05 * selected_instability
        + 0.015 * (
            signal_strength < -95
        ),
        0,
        0.30,
    )

    failed_connection = (
        rng.random(n_rows)
        < failed_connection_probability
    ).astype(int)

    event_ids = [
        f"NET{i:08d}"
        for i in range(1, n_rows + 1)
    ]

    return pd.DataFrame(
        {
            "network_event_id": event_ids,
            "subscriber_id": selected_subscribers,
            "timestamp": timestamps,
            "cell_id": cell_ids,
            "network_type": network_type,
            "signal_strength": signal_strength,
            "latency_ms": latency_ms,
            "packet_loss_pct": packet_loss_pct,
            "data_session_minutes": data_session_minutes,
            "roaming_flag": roaming_flag,
            "failed_connection": failed_connection,
        }
    )


def generate_risk_events(
    config: dict,
    customers: pd.DataFrame,
    behavior_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Generate synthetic risk events."""
    seed = config["data"]["seed"] + 4
    n_rows = config["data"]["risk_events"]["n_rows"]

    rng = np.random.default_rng(seed)

    subscriber_ids = customers["subscriber_id"].to_numpy()

    selected_subscribers = rng.choice(
        subscriber_ids,
        size=n_rows,
        replace=True,
    )

    risk_profile = behavior_profiles.set_index(
        "subscriber_id"
    )["risk_propensity"]

    selected_risk_propensity = risk_profile.loc[
        selected_subscribers
    ].to_numpy()

    start_date = pd.Timestamp(
        config["data"]["time_range"]["start"]
    )

    end_date = pd.Timestamp(
        config["data"]["time_range"]["end"]
    )

    total_seconds = int(
        (end_date - start_date).total_seconds()
    )

    timestamps = start_date + pd.to_timedelta(
        rng.integers(
            0,
            total_seconds,
            size=n_rows,
        ),
        unit="s",
    )

    risk_types = rng.choice(
        [
            "usage_anomaly",
            "network_anomaly",
            "payment_anomaly",
            "account_anomaly",
            "behavioral_anomaly",
            "location_anomaly",
        ],
        size=n_rows,
        p=[
            0.22,
            0.20,
            0.17,
            0.16,
            0.15,
            0.10,
        ],
    )

    base_risk_score = (
        25
        + 55 * selected_risk_propensity
        + rng.normal(
            0,
            10,
            size=n_rows,
        )
    )

    risk_score = np.clip(
        base_risk_score,
        0,
        100,
    )

    risk_level = np.select(
        [
            risk_score >= 80,
            risk_score >= 60,
            risk_score >= 35,
        ],
        [
            "critical",
            "high",
            "medium",
        ],
        default="low",
    )

    trigger_count = np.maximum(
        1,
        rng.poisson(
            1 + 5 * selected_risk_propensity
        ),
    )

    source_system = rng.choice(
        [
            "fraud_engine",
            "network_monitor",
            "billing_system",
            "behavior_engine",
        ],
        size=n_rows,
        p=[
            0.35,
            0.25,
            0.20,
            0.20,
        ],
    )

    event_ids = [
        f"RSK{i:08d}"
        for i in range(1, n_rows + 1)
    ]

    return pd.DataFrame(
        {
            "risk_event_id": event_ids,
            "subscriber_id": selected_subscribers,
            "timestamp": timestamps,
            "risk_type": risk_types,
            "risk_score": np.round(
                risk_score,
                2,
            ),
            "risk_level": risk_level,
            "trigger_count": trigger_count,
            "source_system": source_system,
        }
    )


def save_dataset(
    dataframe: pd.DataFrame,
    filename: str,
) -> None:
    """Save a dataframe to the raw data directory."""
    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = RAW_DATA_DIR / filename

    dataframe.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"Generated {filename}: {dataframe.shape}"
    )
    print(
        f"Saved to: {output_path}"
    )


def main() -> None:
    """Generate all synthetic datasets."""
    config = load_config()

    customers = generate_customers(config)

    behavior_profiles = generate_behavior_profiles(
        customers,
        seed=config["data"]["seed"],
    )

    usage_events = generate_usage_events(
        config,
        customers,
        behavior_profiles,
    )

    recharges = generate_recharges(
        config,
        customers,
        behavior_profiles,
    )

    network_events = generate_network_events(
        config,
        customers,
        behavior_profiles,
    )

    risk_events = generate_risk_events(
        config,
        customers,
        behavior_profiles,
    )

    save_dataset(
        customers,
        "customers.parquet",
    )

    save_dataset(
        usage_events,
        "usage_events.parquet",
    )

    save_dataset(
        recharges,
        "recharges.parquet",
    )

    save_dataset(
        network_events,
        "network_events.parquet",
    )

    save_dataset(
        risk_events,
        "risk_events.parquet",
    )


if __name__ == "__main__":
    main()