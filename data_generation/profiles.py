from __future__ import annotations

import numpy as np
import pandas as pd


def generate_behavior_profiles(
    customers: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_customers = len(customers)

    usage_propensity = rng.beta(
        2.5,
        3.0,
        size=n_customers,
    )

    recharge_propensity = rng.beta(
        2.0,
        3.5,
        size=n_customers,
    )

    network_instability = rng.beta(
        1.8,
        5.0,
        size=n_customers,
    )

    roaming_propensity = rng.beta(
        1.5,
        8.0,
        size=n_customers,
    )

    risk_propensity = (
        0.30 * usage_propensity
        + 0.20 * recharge_propensity
        + 0.20 * network_instability
        + 0.15 * roaming_propensity
        + 0.15 * rng.beta(
            2.0,
            5.0,
            size=n_customers,
        )
    )

    return pd.DataFrame(
        {
            "subscriber_id": customers[
                "subscriber_id"
            ].to_numpy(),
            "usage_propensity": usage_propensity,
            "recharge_propensity": recharge_propensity,
            "network_instability": network_instability,
            "roaming_propensity": roaming_propensity,
            "risk_propensity": np.clip(
                risk_propensity,
                0,
                1,
            ),
        }
    )