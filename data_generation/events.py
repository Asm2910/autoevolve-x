"""Event generators (v2).

Every table keeps the v1 column schema. What changes is the process:

* subscribers differ in activity level and only generate events after activation;
* fraudsters (see ``world.py``) inject extra, type-specific events between
  ``fraud_start`` and ``fraud_end``; a fraction of those events imitates normal
  behaviour, controlled by ``fraud_strength``;
* fraud behaviour drifts over time (``drift_strength``).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Extra events per day, as a multiple of the normal per-subscriber daily rate.
USAGE_MULT = {"sim_box": 5.0, "wangiri": 4.0, "subscription_fraud": 4.0, "account_takeover": 2.0}
RECHARGE_MULT = {"sim_box": 5.0, "subscription_fraud": 4.0, "account_takeover": 3.0, "wangiri": 0.0}
NETWORK_MULT = {"sim_box": 5.0, "account_takeover": 3.0, "subscription_fraud": 0.0, "wangiri": 0.0}
RISK_DETECTION = {"sim_box": 0.9, "wangiri": 0.7, "subscription_fraud": 1.0, "account_takeover": 0.6}

FRAUD_RISK_TYPES = {
    "sim_box": (["usage_anomaly", "network_anomaly", "behavioral_anomaly"], [0.5, 0.3, 0.2]),
    "wangiri": (["usage_anomaly", "behavioral_anomaly", "payment_anomaly"], [0.6, 0.25, 0.15]),
    "subscription_fraud": (["payment_anomaly", "account_anomaly", "usage_anomaly"], [0.45, 0.35, 0.20]),
    "account_takeover": (["account_anomaly", "location_anomaly", "behavioral_anomaly"], [0.40, 0.35, 0.25]),
}


@dataclass
class Context:
    start: pd.Timestamp
    end: pd.Timestamp
    activation_ns: np.ndarray
    start_ns: int
    end_ns: int
    exposure_days: np.ndarray
    drift: float


def build_context(customers: pd.DataFrame, config: dict) -> Context:
    start = pd.Timestamp(config["data"]["time_range"]["start"])
    end = pd.Timestamp(config["data"]["time_range"]["end"])
    activation_ns = pd.DatetimeIndex(customers["activation_date"]).to_numpy().astype("datetime64[ns]").astype("int64")
    start_ns = start.value
    end_ns = end.value
    lo = np.maximum(activation_ns, start_ns)
    exposure_days = np.clip((end_ns - lo) / 86_400e9, 0.5, None)
    return Context(
        start=start,
        end=end,
        activation_ns=activation_ns,
        start_ns=start_ns,
        end_ns=end_ns,
        exposure_days=exposure_days,
        drift=float(config["data"]["fraud"]["drift_strength"]),
    )


def _uniform_ns(rng, lo_ns, hi_ns):
    lo_ns = np.asarray(lo_ns, dtype="int64")
    hi_ns = np.asarray(hi_ns, dtype="int64")
    return lo_ns + (rng.random(len(lo_ns)) * np.maximum(hi_ns - lo_ns, 0)).astype("int64")


def _baseline_draw(rng, ctx: Context, weights: np.ndarray, k: int):
    probability = weights / weights.sum()
    position = rng.choice(len(weights), size=k, replace=True, p=probability)
    lo = np.maximum(ctx.activation_ns[position], ctx.start_ns)
    ts_ns = _uniform_ns(rng, lo, np.full(k, ctx.end_ns))
    return position, ts_ns


@dataclass
class FraudEvents:
    position: np.ndarray
    ts_ns: np.ndarray
    fraud_type: np.ndarray
    strength: np.ndarray
    pure: np.ndarray


def _fraud_events(rng, world: pd.DataFrame, multipliers: dict, base_daily: float) -> FraudEvents:
    fraud = world[world["is_fraud"] == 1]
    position = fraud.index.to_numpy()
    start_ns = fraud["fraud_start"].to_numpy().astype("datetime64[ns]").astype("int64")
    end_ns = fraud["fraud_end"].to_numpy().astype("datetime64[ns]").astype("int64")
    days = np.clip((end_ns - start_ns) / 86_400e9, 0.5, None)
    strength = fraud["fraud_strength"].to_numpy()
    ftype = fraud["fraud_type"].to_numpy()
    mult = np.array([multipliers[t] for t in ftype])

    lam = base_daily * mult * (0.3 + 1.4 * strength) * days
    counts = rng.poisson(lam)
    rep = np.repeat(np.arange(len(fraud)), counts)
    ts_ns = _uniform_ns(rng, start_ns[rep], end_ns[rep])
    pure = rng.random(len(rep)) < (0.35 + 0.65 * strength[rep])
    return FraudEvents(position[rep], ts_ns, ftype[rep], strength[rep], pure)


def _finish(frame: pd.DataFrame, id_column: str, prefix: str) -> pd.DataFrame:
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    frame.insert(0, id_column, [f"{prefix}{i:08d}" for i in range(1, len(frame) + 1)])
    return frame


def _drift_fraction(ctx: Context, ts_ns: np.ndarray) -> np.ndarray:
    return np.clip((ts_ns - ctx.start_ns) / (ctx.end_ns - ctx.start_ns), 0, 1)


# --------------------------------------------------------------------------
# Usage events
# --------------------------------------------------------------------------
def generate_usage_events(config, customers, profiles, world, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ctx = build_context(customers, config)
    n_rows = config["data"]["usage_events"]["n_rows"]
    sid = customers["subscriber_id"].to_numpy()

    usage_p = profiles["usage_propensity"].to_numpy()
    base_daily = n_rows / ctx.exposure_days.sum()

    fraud = _fraud_events(rng, world, USAGE_MULT, base_daily)
    n_base = n_rows - len(fraud.position)
    weights = profiles["activity_level"].to_numpy() * (0.5 + usage_p) * ctx.exposure_days
    base_pos, base_ts = _baseline_draw(rng, ctx, weights, n_base)

    pos = np.concatenate([base_pos, fraud.position])
    ts_ns = np.concatenate([base_ts, fraud.ts_ns])
    pure = np.concatenate([np.zeros(n_base, dtype=bool), fraud.pure])
    ftype = np.concatenate([np.full(n_base, "none", dtype=object), fraud.fraud_type])
    s = np.concatenate([np.zeros(n_base), fraud.strength])
    n = len(pos)
    t_frac = _drift_fraction(ctx, ts_ns)

    intensity = usage_p[pos]
    intl_t = profiles["roaming_propensity"].to_numpy()[pos]
    night_t = profiles["night_propensity"].to_numpy()[pos]
    dest_t = profiles["destination_propensity"].to_numpy()[pos]
    heavy = (profiles["lookalike"].to_numpy()[pos] == "heavy_caller")

    call_lambda = 2 + 12 * intensity
    sms_lambda = 1 + 8 * intensity
    short_p = np.clip(0.10 + 0.20 * dest_t + 0.15 * intl_t + 0.12 * heavy, 0, 0.95)
    intl_p = np.clip(0.02 + 0.30 * intl_t, 0, 1)
    dest_lambda = 1 + 8 * dest_t
    night_p = np.clip(0.08 + 0.45 * night_t, 0, 0.95)
    roam_p = np.clip(0.05 + 0.10 * intl_t, 0, 1)
    data_scale = 0.5 + intensity
    duration_scale = np.ones(n)

    def override(kind):
        return pure & (ftype == kind)

    m = override("sim_box")
    call_lambda[m] = 12 + 14 * s[m]
    short_p[m] = np.clip(0.72 - 0.32 * ctx.drift * t_frac[m] + 0.10 * (s[m] - 0.5), 0.2, 0.95)
    intl_p[m] = 0.45
    dest_lambda[m] = 14 + 22 * s[m]
    night_p[m] = 0.70
    sms_lambda[m] = 0.3
    data_scale[m] = 0.1
    roam_p[m] = 0.03

    m = override("wangiri")
    call_lambda[m] = 8 + 10 * s[m]
    short_p[m] = 0.92
    intl_p[m] = 0.85
    dest_lambda[m] = 16 + 20 * s[m]
    night_p[m] = 0.45
    sms_lambda[m] = 0.5
    data_scale[m] = 0.3

    m = override("subscription_fraud")
    call_lambda[m] = 10 + 12 * s[m]
    short_p[m] = 0.25
    intl_p[m] = 0.45
    dest_lambda[m] = 7 + 9 * s[m]
    night_p[m] = 0.40
    duration_scale[m] = 2.0

    m = override("account_takeover")
    call_lambda[m] = 5 + 3 * s[m]
    sms_lambda[m] = 4 + 6 * s[m]
    data_scale[m] = 1.5 + 2 * s[m]
    roam_p[m] = 0.50
    intl_p[m] = 0.25
    night_p[m] = 0.55
    dest_lambda[m] = 4 + 4 * s[m]

    call_count = rng.poisson(call_lambda)
    call_duration = call_count * rng.gamma(2.0, 45.0, n) * duration_scale
    sms_count = rng.poisson(sms_lambda)
    data_mb = rng.gamma(2.0, 150.0, n) * data_scale
    international_calls = np.where(call_count > 0, rng.binomial(call_count, intl_p), 0)
    short_calls = np.where(call_count > 0, rng.binomial(call_count, short_p), 0)
    unique_destinations = np.maximum(1, rng.poisson(dest_lambda))
    night_activity = rng.binomial(1, night_p)
    roaming_flag = rng.binomial(1, roam_p)

    frame = pd.DataFrame(
        {
            "subscriber_id": sid[pos],
            "timestamp": pd.to_datetime(ts_ns),
            "call_count": call_count,
            "call_duration_sec": call_duration,
            "sms_count": sms_count,
            "data_mb": data_mb,
            "international_calls": international_calls,
            "short_calls": short_calls,
            "unique_destinations": unique_destinations,
            "night_activity": night_activity,
            "roaming_flag": roaming_flag,
        }
    )
    return _finish(frame, "event_id", "EVT")


# --------------------------------------------------------------------------
# Recharges
# --------------------------------------------------------------------------
def generate_recharges(config, customers, profiles, world, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ctx = build_context(customers, config)
    n_rows = config["data"]["recharges"]["n_rows"]
    sid = customers["subscriber_id"].to_numpy()

    prepaid_all = (customers["plan_type"] == "prepaid").to_numpy()
    rech_p = profiles["recharge_propensity"].to_numpy()
    base_daily = n_rows / ctx.exposure_days.sum()

    fraud = _fraud_events(rng, world, RECHARGE_MULT, base_daily)
    n_base = n_rows - len(fraud.position)
    weights = (
        profiles["activity_level"].to_numpy()
        * np.where(prepaid_all, 2.0, 1.0)
        * (0.5 + rech_p)
        * ctx.exposure_days
    )
    base_pos, base_ts = _baseline_draw(rng, ctx, weights, n_base)

    pos = np.concatenate([base_pos, fraud.position])
    ts_ns = np.concatenate([base_ts, fraud.ts_ns])
    pure = np.concatenate([np.zeros(n_base, dtype=bool), fraud.pure])
    ftype = np.concatenate([np.full(n_base, "none", dtype=object), fraud.fraud_type])
    n = len(pos)

    prepaid = prepaid_all[pos]
    segment = customers["customer_segment"].to_numpy()[pos]
    propensity = rech_p[pos]
    bursty = profiles["lookalike"].to_numpy()[pos] == "bursty_recharger"

    amount = np.where(
        prepaid,
        rng.lognormal(np.log(12), 0.55, n),
        rng.lognormal(np.log(45), 0.45, n),
    )
    amount *= np.where(segment == "business", 1.25, 1.0)
    amount *= np.where(segment == "premium", 1.50, 1.0)
    amount *= 0.70 + 0.90 * propensity

    channel_score = rng.random(n)
    channel = np.where(
        channel_score < 0.20 + 0.25 * propensity,
        "mobile_app",
        np.where(
            channel_score < 0.45 + 0.15 * propensity,
            "ussd",
            np.where(channel_score < 0.70, "retail_store", np.where(channel_score < 0.90, "bank", "web")),
        ),
    )

    type_draw = rng.random(n)
    prepaid_prob = np.clip(0.55 + 0.25 * propensity - 0.20 * bursty, 0, 1)
    emergency_cut = np.where(bursty, 0.65, 0.90)
    recharge_type = np.where(
        prepaid,
        np.where(type_draw < prepaid_prob, "standard", np.where(type_draw < emergency_cut, "bundle", "emergency")),
        np.where(rng.random(n) < 0.85, "bill_payment", "advance_payment"),
    )

    fail_p = np.where(channel == "retail_store", 0.04, 0.02) * (1.20 - 0.40 * propensity)
    fail_p = np.clip(fail_p, 0.005, 0.10)

    # fraud overrides
    m = pure & (ftype == "sim_box")
    amount[m] = rng.lognormal(np.log(5), 0.4, m.sum())
    channel[m] = rng.choice(["retail_store", "ussd", "bank"], m.sum(), p=[0.5, 0.4, 0.1])
    recharge_type[m] = np.where(
        prepaid[m], rng.choice(["bundle", "standard"], m.sum(), p=[0.6, 0.4]), "bill_payment"
    )
    fail_p[m] = 0.03

    m = pure & (ftype == "subscription_fraud")
    amount[m] = rng.lognormal(np.log(25), 0.5, m.sum())
    channel[m] = rng.choice(["web", "mobile_app"], m.sum(), p=[0.6, 0.4])
    recharge_type[m] = np.where(prepaid[m], "standard", "advance_payment")
    fail_p[m] = 0.60

    m = pure & (ftype == "account_takeover")
    amount[m] = amount[m] * 3.5
    channel[m] = rng.choice(["mobile_app", "web"], m.sum(), p=[0.5, 0.5])
    recharge_type[m] = np.where(
        prepaid[m], rng.choice(["emergency", "standard"], m.sum(), p=[0.7, 0.3]), "advance_payment"
    )
    fail_p[m] = 0.20

    successful = (rng.random(n) > fail_p).astype(int)

    frame = pd.DataFrame(
        {
            "subscriber_id": sid[pos],
            "timestamp": pd.to_datetime(ts_ns),
            "recharge_amount": np.round(amount, 2),
            "payment_channel": channel,
            "recharge_type": recharge_type,
            "successful": successful,
        }
    )
    return _finish(frame, "recharge_id", "RCH")


# --------------------------------------------------------------------------
# Network events
# --------------------------------------------------------------------------
def generate_network_events(config, customers, profiles, world, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ctx = build_context(customers, config)
    n_rows = config["data"]["network_events"]["n_rows"]
    sid = customers["subscriber_id"].to_numpy()

    usage_p = profiles["usage_propensity"].to_numpy()
    base_daily = n_rows / ctx.exposure_days.sum()

    fraud = _fraud_events(rng, world, NETWORK_MULT, base_daily)
    n_base = n_rows - len(fraud.position)
    weights = profiles["activity_level"].to_numpy() * (0.5 + usage_p) * ctx.exposure_days
    base_pos, base_ts = _baseline_draw(rng, ctx, weights, n_base)

    pos = np.concatenate([base_pos, fraud.position])
    ts_ns = np.concatenate([base_ts, fraud.ts_ns])
    pure = np.concatenate([np.zeros(n_base, dtype=bool), fraud.pure])
    ftype = np.concatenate([np.full(n_base, "none", dtype=object), fraud.fraud_type])
    n = len(pos)

    instability = profiles["network_instability"].to_numpy()[pos]
    home_cell = profiles["home_cell"].to_numpy()[pos]
    roam_t = profiles["roaming_propensity"].to_numpy()[pos]

    at_home = rng.random(n) < 0.70
    cell = np.where(at_home, home_cell, rng.integers(1, 501, n))
    network_type = rng.choice(["3G", "4G", "5G"], n, p=[0.10, 0.55, 0.35])

    signal = rng.normal(-75, 10, n) - 25 * instability
    latency = rng.gamma(2.0, 15.0, n) * (1.0 + 1.5 * instability)
    packet_loss = rng.beta(2.0, 5.0, n) * 2.5 * (1.0 + 2.0 * instability)
    session = rng.gamma(2.0, 8.0, n) * (0.7 + 0.8 * (1 - instability))
    roam_p = np.clip(0.04 + 0.08 * instability + 0.25 * roam_t, 0, 1)

    m = pure & (ftype == "sim_box")
    second_cell = (home_cell % 500) + 1
    cell[m] = np.where(rng.random(m.sum()) < 0.7, home_cell[m], second_cell[m])
    signal[m] = rng.normal(-66, 4, m.sum())
    session[m] = rng.gamma(1.5, 3.0, m.sum())
    latency[m] = latency[m] * 0.8
    roam_p[m] = 0.01
    network_type[m] = rng.choice(["3G", "4G", "5G"], m.sum(), p=[0.10, 0.70, 0.20])

    m = pure & (ftype == "account_takeover")
    cell[m] = rng.integers(1, 501, m.sum())
    roam_p[m] = 0.55
    network_type[m] = rng.choice(["3G", "4G", "5G"], m.sum(), p=[0.35, 0.50, 0.15])

    signal = np.clip(signal, -120, -40)
    fail_p = np.clip(0.008 + 0.05 * instability + 0.015 * (signal < -95), 0, 0.30)

    frame = pd.DataFrame(
        {
            "subscriber_id": sid[pos],
            "timestamp": pd.to_datetime(ts_ns),
            "cell_id": [f"CELL_{c:05d}" for c in cell],
            "network_type": network_type,
            "signal_strength": signal,
            "latency_ms": latency,
            "packet_loss_pct": packet_loss,
            "data_session_minutes": session,
            "roaming_flag": rng.binomial(1, roam_p),
            "failed_connection": (rng.random(n) < fail_p).astype(int),
        }
    )
    return _finish(frame, "network_event_id", "NET")


# --------------------------------------------------------------------------
# Risk events (output of a noisy rule engine - independent evidence, not the label)
# --------------------------------------------------------------------------
def generate_risk_events(config, customers, profiles, world, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ctx = build_context(customers, config)
    n_rows = config["data"]["risk_events"]["n_rows"]
    sid = customers["subscriber_id"].to_numpy()

    # fraud-driven alerts
    fraud = world[world["is_fraud"] == 1]
    f_pos = fraud.index.to_numpy()
    f_start = fraud["fraud_start"].to_numpy().astype("datetime64[ns]").astype("int64")
    f_end = fraud["fraud_end"].to_numpy().astype("datetime64[ns]").astype("int64")
    f_days = np.clip((f_end - f_start) / 86_400e9, 0.5, None)
    f_s = fraud["fraud_strength"].to_numpy()
    f_type = fraud["fraud_type"].to_numpy()
    detect = np.array([RISK_DETECTION[t] for t in f_type])
    counts = rng.poisson((f_days / 30.0) * detect * (0.3 + 1.7 * f_s))
    rep = np.repeat(np.arange(len(fraud)), counts)
    fraud_ts = _uniform_ns(rng, f_start[rep], f_end[rep])
    fraud_pos = f_pos[rep]
    fraud_s = f_s[rep]
    fraud_kind = f_type[rep]
    n_fraud = len(rep)

    # baseline alerts (false positives, more for lookalikes)
    n_base = n_rows - n_fraud
    lookalike = profiles["lookalike"].to_numpy() != "none"
    weights = profiles["alert_rate"].to_numpy() * (1 + 2.0 * lookalike) * ctx.exposure_days
    base_pos, base_ts = _baseline_draw(rng, ctx, weights, n_base)

    pos = np.concatenate([base_pos, fraud_pos])
    ts_ns = np.concatenate([base_ts, fraud_ts])
    is_fraud_alert = np.concatenate([np.zeros(n_base, dtype=bool), np.ones(n_fraud, dtype=bool)])
    strength = np.concatenate([np.zeros(n_base), fraud_s])
    kind = np.concatenate([np.full(n_base, "none", dtype=object), fraud_kind])
    n = len(pos)

    risk_type = rng.choice(
        [
            "usage_anomaly", "network_anomaly", "payment_anomaly",
            "account_anomaly", "behavioral_anomaly", "location_anomaly",
        ],
        n,
        p=[0.22, 0.20, 0.17, 0.16, 0.15, 0.10],
    ).astype(object)
    score = rng.normal(38, 12, n) + 14 * lookalike[pos]
    trigger = 1 + rng.poisson(0.6, n)

    for fraud_kind_name, (types, probs) in FRAUD_RISK_TYPES.items():
        m = is_fraud_alert & (kind == fraud_kind_name)
        risk_type[m] = rng.choice(types, m.sum(), p=probs)
    m = is_fraud_alert
    score[m] = rng.normal(55 + 22 * strength[m], 12)
    trigger[m] = 1 + rng.poisson(2 + 4 * strength[m])

    score = np.clip(score, 0, 100)
    level = np.select(
        [score >= 80, score >= 60, score >= 35],
        ["critical", "high", "medium"],
        default="low",
    )
    source = rng.choice(
        ["fraud_engine", "network_monitor", "billing_system", "behavior_engine"],
        n,
        p=[0.35, 0.25, 0.20, 0.20],
    )

    frame = pd.DataFrame(
        {
            "subscriber_id": sid[pos],
            "timestamp": pd.to_datetime(ts_ns),
            "risk_type": risk_type,
            "risk_score": np.round(score, 2),
            "risk_level": level,
            "trigger_count": trigger,
            "source_system": source,
        }
    )
    return _finish(frame, "risk_event_id", "RSK")
