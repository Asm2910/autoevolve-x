"""Hidden ground truth: who is a fraudster, what kind, and when.

The fraud label is decided HERE, before any event is generated. Events are then
generated conditional on this state, so the label is never a function of the
features (the v1 circularity flaw).
"""

import numpy as np
import pandas as pd

FRAUD_TYPES = ["sim_box", "subscription_fraud", "account_takeover", "wangiri"]

REGION_RISK = {"north": 1.0, "south": 1.3, "east": 0.8, "west": 1.0, "central": 1.1}

DAY_NS = 86_400 * 10**9


def _ns(values) -> np.ndarray:
    return pd.DatetimeIndex(values).to_numpy().astype("datetime64[ns]").astype("int64")


def _selection_probabilities(customers: pd.DataFrame, window_start: pd.Timestamp):
    """Base-rate differences: some customers are more likely to be targeted."""
    weight = np.ones(len(customers))
    weight *= np.where(customers["plan_type"] == "prepaid", 1.8, 1.0)
    weight *= np.where(customers["device_type"] == "feature_phone", 1.3, 1.0)
    weight *= np.where(customers["customer_segment"] == "business", 0.7, 1.0)
    weight *= customers["region"].map(REGION_RISK).to_numpy()
    recent = customers["activation_date"] >= (window_start - pd.Timedelta(days=90))
    weight *= np.where(recent, 2.5, 1.0)
    return weight / weight.sum()


def build_world(customers: pd.DataFrame, config: dict, seed: int) -> pd.DataFrame:
    """Return one row per subscriber with hidden fraud state."""
    rng = np.random.default_rng(seed)
    data_cfg = config["data"]
    fraud_cfg = data_cfg["fraud"]

    start = pd.Timestamp(data_cfg["time_range"]["start"])
    end = pd.Timestamp(data_cfg["time_range"]["end"])
    start_ns, end_ns = start.value, end.value
    n = len(customers)

    n_fraud = int(round(n * fraud_cfg["rate"]))
    probs = _selection_probabilities(customers, start)
    fraud_idx = np.sort(rng.choice(n, size=n_fraud, replace=False, p=probs))
    activation_ns = _ns(customers["activation_date"])[fraud_idx]

    lo = start_ns + 45 * DAY_NS
    hi = end_ns - 15 * DAY_NS
    fraud_start = lo + (rng.random(n_fraud) * (hi - lo)).astype("int64")
    # fraud cannot start before the line exists
    fraud_start = np.clip(np.maximum(fraud_start, activation_ns + DAY_NS), lo, hi)

    # Fraud type mix drifts over time (concept drift).
    t_frac = (fraud_start - start_ns) / (end_ns - start_ns)
    mix_start = np.array(fraud_cfg["type_mix_start"], dtype=float)
    mix_end = np.array(fraud_cfg["type_mix_end"], dtype=float)
    mix = (1 - t_frac)[:, None] * mix_start + t_frac[:, None] * mix_end

    # Subscription fraud only happens on newly activated lines.
    is_new = (fraud_start - activation_ns) < 180 * DAY_NS
    mix[:, 1] *= np.where(is_new, 6.0, 0.0)
    mix = mix / mix.sum(axis=1, keepdims=True)
    draw = rng.random(n_fraud)[:, None]
    type_idx = (draw > mix.cumsum(axis=1)).sum(axis=1).clip(0, len(FRAUD_TYPES) - 1)
    fraud_type = np.array(FRAUD_TYPES, dtype=object)[type_idx]

    # Subscription fraud starts shortly after activation.
    sub = fraud_type == "subscription_fraud"
    sub_start = activation_ns[sub] + (rng.uniform(1, 21, sub.sum()) * DAY_NS).astype("int64")
    fraud_start[sub] = np.clip(sub_start, lo, hi)

    strength = rng.beta(2.2, 2.0, n_fraud)

    duration_ns = (
        rng.lognormal(np.log(fraud_cfg["duration_median_days"]), 0.6, n_fraud) * DAY_NS
    ).astype("int64")
    confirmed = rng.random(n_fraud) < fraud_cfg["confirm_probability"]
    lag_ns = (
        (2.0 + rng.lognormal(np.log(fraud_cfg["confirm_lag_median_days"]), 0.7, n_fraud))
        * DAY_NS
    ).astype("int64")
    confirm_ns = fraud_start + lag_ns

    # Behaviour stops when the line is blocked (confirmation) or the campaign ends.
    natural_end = fraud_start + duration_ns
    behaviour_end = np.where(confirmed, np.minimum(natural_end, confirm_ns), natural_end)
    behaviour_end = np.minimum(behaviour_end, end_ns)

    nat = np.datetime64("NaT")
    fraud_start_col = np.full(n, nat, dtype="datetime64[ns]")
    fraud_end_col = np.full(n, nat, dtype="datetime64[ns]")
    confirm_col = np.full(n, nat, dtype="datetime64[ns]")
    fraud_start_col[fraud_idx] = fraud_start.astype("datetime64[ns]")
    fraud_end_col[fraud_idx] = behaviour_end.astype("datetime64[ns]")
    confirm_col[fraud_idx] = np.where(confirmed, confirm_ns, np.iinfo("int64").min).astype(
        "datetime64[ns]"
    )

    is_fraud = np.zeros(n, dtype=int)
    is_fraud[fraud_idx] = 1
    type_col = np.full(n, "none", dtype=object)
    type_col[fraud_idx] = fraud_type
    strength_col = np.zeros(n)
    strength_col[fraud_idx] = strength

    # Label noise: occasional wrong confirmations on innocent subscribers.
    innocent = np.setdiff1d(np.arange(n), fraud_idx)
    n_false = int(round(len(innocent) * fraud_cfg["false_confirm_rate"]))
    false_idx = rng.choice(innocent, size=n_false, replace=False)
    false_lo = start_ns + 60 * DAY_NS
    false_dates = false_lo + (rng.random(n_false) * (end_ns - false_lo)).astype("int64")
    confirm_col[false_idx] = false_dates.astype("datetime64[ns]")
    false_flag = np.zeros(n, dtype=int)
    false_flag[false_idx] = 1

    return pd.DataFrame(
        {
            "subscriber_id": customers["subscriber_id"].to_numpy(),
            "is_fraud": is_fraud,
            "fraud_type": type_col,
            "fraud_strength": strength_col,
            "fraud_start": fraud_start_col,
            "fraud_end": fraud_end_col,
            "confirm_date": confirm_col,
            "false_confirmation": false_flag,
        }
    )
