"""Hidden behaviour profiles for *legitimate* subscriber behaviour.

v2 change: profiles are independent of fraud. There is no blended
"risk_propensity" any more - fraud lives in ``world.py`` and is the only thing
that creates fraud-like behaviour.
"""

import numpy as np
import pandas as pd

LOOKALIKES = ["heavy_caller", "traveller", "night_worker", "bursty_recharger"]


def generate_behavior_profiles(
    customers: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Generate per-subscriber normal-behaviour parameters."""
    rng = np.random.default_rng(seed)
    n = len(customers)

    segment = customers["customer_segment"].to_numpy()
    business = segment == "business"
    premium = segment == "premium"

    usage = rng.beta(2.0, 4.0, n)
    usage = np.where(business, np.clip(usage * 1.25 + 0.08, 0, 1), usage)
    usage = np.where(premium, np.clip(usage * 1.15 + 0.05, 0, 1), usage)

    recharge = rng.beta(2.0, 3.0, n)
    network = rng.beta(2.0, 6.0, n)
    roaming = np.where(premium, rng.beta(2.0, 5.0, n), rng.beta(1.5, 8.0, n))
    destination = np.where(business, rng.beta(3.0, 4.0, n), rng.beta(2.0, 5.0, n))
    night = rng.beta(2.0, 6.0, n)
    alert_rate = rng.beta(1.2, 10.0, n)
    activity = rng.lognormal(0.0, 0.6, n)
    home_cell = rng.integers(1, 501, n)

    # Legitimate subscribers that look unusual (they are NOT fraud).
    lookalike = np.full(n, "none", dtype=object)
    draw = rng.random(n)
    lookalike[draw < 0.04] = "heavy_caller"
    lookalike[(draw >= 0.04) & (draw < 0.08)] = "traveller"
    lookalike[(draw >= 0.08) & (draw < 0.11)] = "night_worker"
    lookalike[(draw >= 0.11) & (draw < 0.14)] = "bursty_recharger"

    heavy = lookalike == "heavy_caller"
    usage = np.where(heavy, np.maximum(usage, 0.80), usage)
    destination = np.where(heavy, np.maximum(destination, 0.65), destination)

    traveller = lookalike == "traveller"
    roaming = np.where(traveller, np.maximum(roaming, 0.60), roaming)

    night_worker = lookalike == "night_worker"
    night = np.where(night_worker, np.maximum(night, 0.70), night)

    bursty = lookalike == "bursty_recharger"
    recharge = np.where(bursty, np.maximum(recharge, 0.85), recharge)

    return pd.DataFrame(
        {
            "subscriber_id": customers["subscriber_id"].to_numpy(),
            "usage_propensity": usage,
            "recharge_propensity": recharge,
            "network_instability": network,
            "roaming_propensity": roaming,
            "destination_propensity": destination,
            "night_propensity": night,
            "alert_rate": alert_rate,
            "activity_level": activity,
            "home_cell": home_cell,
            "lookalike": lookalike,
        }
    )
