"""Snapshot feature builder.

For a snapshot date T, every feature is computed from events in (T - window, T]
only, so nothing from the future can leak into a row. Column names follow

    <group>__<metric>__<window>d      e.g. usage__total_calls__30d

so a feature's group and window can be read straight from its name. Customer
features have no window: customer__<metric>.
"""

from __future__ import annotations

import pandas as pd

from feature_engineering.cells import build_cell_features
from feature_engineering.customer import build_customer_features
from feature_engineering.network import build_network_features
from feature_engineering.recharges import build_recharge_features
from feature_engineering.risk import build_risk_features
from feature_engineering.usage import build_usage_features

WINDOWS = (7, 30, 90)
SEP = "__"
ID_COLUMNS = ["subscriber_id", "snapshot_date"]
LABEL = "risk_label"

# Metrics whose natural value is 0 when a subscriber has no events in the window.
COUNT_PREFIXES = (
    "total_",
    "successful_",
    "failed_",
    "high_risk_event_count",
    "critical_risk_event_count",
    "unique_cells",
)

# (group, metric) pairs for which we add "recent vs long-run" activity ratios.
TREND_METRICS = [
    ("usage", "total_calls"),
    ("usage", "total_international_calls"),
    ("usage", "total_short_calls"),
    ("recharge", "total_recharges"),
    ("network", "total_network_events"),
    ("risk", "total_risk_events"),
]


def _build_network_group(events: pd.DataFrame) -> pd.DataFrame:
    base = build_network_features(events)
    cells = build_cell_features(events)
    return base.merge(cells, on="subscriber_id", how="left")


# group name -> (dataset key, builder function)
GROUP_BUILDERS = {
    "usage": ("usage", build_usage_features),
    "recharge": ("recharges", build_recharge_features),
    "network": ("network", _build_network_group),
    "risk": ("risk", build_risk_features),
}


def _in_window(events: pd.DataFrame, as_of: pd.Timestamp, days: int) -> pd.DataFrame:
    lower = as_of - pd.Timedelta(days=days)
    stamp = events["timestamp"]
    return events[(stamp > lower) & (stamp <= as_of)]


def _customer_block(customers: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Customer attributes as known on the snapshot date."""
    existing = customers[customers["activation_date"] <= as_of].copy()
    existing["tenure_days"] = (as_of - existing["activation_date"]).dt.days

    features = build_customer_features(existing)
    redundant = [c for c in features.columns if c.startswith("is_")]
    features = features.drop(columns=redundant)

    return features.rename(
        columns={c: f"customer{SEP}{c}" for c in features.columns if c != "subscriber_id"}
    )


def build_snapshot_features(
    customers: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    snapshot_rows: pd.DataFrame,
    as_of: pd.Timestamp,
    windows: tuple[int, ...] = WINDOWS,
) -> pd.DataFrame:
    """Build features for every subscriber in ``snapshot_rows`` as of ``as_of``."""
    as_of = pd.Timestamp(as_of)
    longest = max(windows)

    recent = {
        key: _in_window(datasets[key], as_of, longest)
        for key in {table for table, _ in GROUP_BUILDERS.values()}
    }

    base = snapshot_rows[["subscriber_id"]].merge(
        _customer_block(customers, as_of),
        on="subscriber_id",
        how="left",
        validate="one_to_one",
    )
    index = pd.Index(base["subscriber_id"], name="subscriber_id")

    pieces = [base.set_index("subscriber_id")]
    flags: dict[str, pd.Series] = {}

    for days in windows:
        for group, (table, builder) in GROUP_BUILDERS.items():
            flag_name = f"{group}{SEP}has_events{SEP}{days}d"
            window_events = _in_window(recent[table], as_of, days)

            if window_events.empty:
                flags[flag_name] = pd.Series(0, index=index)
                continue

            block = builder(window_events).set_index("subscriber_id")
            block.columns = [f"{group}{SEP}{c}{SEP}{days}d" for c in block.columns]
            flags[flag_name] = pd.Series(index.isin(block.index), index=index)
            pieces.append(block.reindex(index))

    pieces.append(pd.DataFrame(flags))
    frame = pd.concat(pieces, axis=1).copy().reset_index().assign(snapshot_date=as_of)
    return frame.merge(
        snapshot_rows[["subscriber_id", LABEL]],
        on="subscriber_id",
        how="left",
        validate="one_to_one",
    )


def fill_missing_features(features: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps honestly.

    * counts / sums  -> 0 when there were no events
    * type shares    -> 0 when there were events but none of that type
    * averages, min, max, std -> left missing (no events means no value);
      the ``has_events`` flags tell the model that the value is missing.
    """
    result = features.copy()

    for column in result.columns:
        parts = column.split(SEP)
        if len(parts) != 3:
            continue
        group, metric, window = parts

        if metric == "has_events":
            result[column] = result[column].fillna(False).astype("int8")
        elif metric.startswith(COUNT_PREFIXES):
            result[column] = result[column].fillna(0)
        elif "_rate_" in metric:
            has_events = result[f"{group}{SEP}has_events{SEP}{window}"] == 1
            result.loc[has_events, column] = result.loc[has_events, column].fillna(0)

    return result


def add_trend_features(features: pd.DataFrame) -> pd.DataFrame:
    """Ratios of recent daily activity to long-run daily activity."""
    result = features.copy()

    for group, metric in TREND_METRICS:
        long_rate = (result[f"{group}{SEP}{metric}{SEP}90d"] / 90).clip(lower=1e-3)
        for days in (7, 30):
            short_rate = result[f"{group}{SEP}{metric}{SEP}{days}d"] / days
            name = f"{group}{SEP}{metric}_trend{SEP}{days}d_vs_90d"
            result[name] = short_rate / long_rate

    return result


def build_feature_dataset(
    customers: pd.DataFrame,
    usage_events: pd.DataFrame,
    recharges: pd.DataFrame,
    network_events: pd.DataFrame,
    risk_events: pd.DataFrame,
    snapshot_labels: pd.DataFrame,
    windows: tuple[int, ...] = WINDOWS,
) -> pd.DataFrame:
    """Build the modelling dataset: one row per (subscriber, snapshot)."""
    datasets = {
        "usage": usage_events,
        "recharges": recharges,
        "network": network_events,
        "risk": risk_events,
    }

    frames = []
    for as_of, rows in snapshot_labels.groupby("snapshot_date"):
        print(f"  snapshot {pd.Timestamp(as_of).date()}  rows={len(rows):,}", flush=True)
        frames.append(build_snapshot_features(customers, datasets, rows, as_of, windows))

    features = pd.concat(frames, ignore_index=True)
    features = fill_missing_features(features)
    features = add_trend_features(features)

    feature_columns = [c for c in features.columns if c not in (*ID_COLUMNS, LABEL)]
    float_columns = features[feature_columns].select_dtypes("float64").columns
    features[float_columns] = features[float_columns].astype("float32")

    return features[[*ID_COLUMNS, *feature_columns, LABEL]]


def feature_manifest(features: pd.DataFrame) -> dict:
    """Describe the feature columns by group, so experiments can select them."""
    feature_columns = [c for c in features.columns if c not in (*ID_COLUMNS, LABEL)]
    groups: dict[str, list[str]] = {}
    for column in feature_columns:
        groups.setdefault(column.split(SEP)[0], []).append(column)

    categorical = [
        c
        for c in feature_columns
        if not pd.api.types.is_numeric_dtype(features[c])
    ]
    return {
        "id_columns": ID_COLUMNS,
        "label": LABEL,
        "features": feature_columns,
        "categorical": categorical,
        "groups": groups,
    }
