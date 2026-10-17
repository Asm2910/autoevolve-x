"""Snapshot index and labels.

A snapshot is "score every active subscriber on date T". The label asks:
will fraud be CONFIRMED in the next ``horizon_days``?  Subscribers already
confirmed before T are excluded (already known).
"""

import pandas as pd


def snapshot_dates(config: dict) -> pd.DatetimeIndex:
    cfg = config["data"]["snapshots"]
    return pd.date_range(cfg["first"], cfg["last"], freq="ME")


def build_snapshot_labels(customers: pd.DataFrame, world: pd.DataFrame, config: dict) -> pd.DataFrame:
    horizon = pd.Timedelta(days=config["data"]["snapshots"]["horizon_days"])
    base = customers[["subscriber_id", "activation_date"]].merge(
        world[["subscriber_id", "confirm_date"]], on="subscriber_id", how="left"
    )
    frames = []
    for snapshot in snapshot_dates(config):
        active = base["activation_date"] <= snapshot
        already_known = base["confirm_date"].notna() & (base["confirm_date"] <= snapshot)
        eligible = base[active & ~already_known]
        label = (
            eligible["confirm_date"].notna()
            & (eligible["confirm_date"] > snapshot)
            & (eligible["confirm_date"] <= snapshot + horizon)
        ).astype(int)
        frames.append(
            pd.DataFrame(
                {
                    "subscriber_id": eligible["subscriber_id"].to_numpy(),
                    "snapshot_date": snapshot,
                    "risk_label": label.to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
