"""Cell-tower features: how many cells a subscriber uses and how concentrated."""

import pandas as pd


def build_cell_features(network_events: pd.DataFrame) -> pd.DataFrame:
    """Return unique_cells and top_cell_share per subscriber.

    A subscriber that sits on one or two cells all day (a SIM-box) has a low
    unique_cells and a high top_cell_share; a normal mobile user does not.
    """
    per_cell = network_events.groupby(["subscriber_id", "cell_id"]).size()
    per_subscriber = per_cell.groupby("subscriber_id")

    features = pd.DataFrame(
        {
            "unique_cells": per_subscriber.size(),
            "top_cell_share": per_subscriber.max() / per_subscriber.sum(),
        }
    )
    return features.reset_index()
