"""Generate the v2 synthetic telecom dataset.

Run with:  python -m scripts.generate_data
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data_generation import events
from data_generation.profiles import generate_behavior_profiles
from data_generation.snapshots import build_snapshot_labels
from data_generation.world import build_world

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
HIDDEN_DATA_DIR = PROJECT_ROOT / "data" / "hidden"


def load_config() -> dict:
    """Load project data-generation configuration."""
    with open(CONFIG_PATH, encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_customers(config: dict) -> pd.DataFrame:
    """Generate synthetic customer master data."""
    data_cfg = config["data"]
    n_rows = data_cfg["customers"]["n_rows"]
    rng = np.random.default_rng(data_cfg["seed"])

    activation_start = pd.Timestamp(data_cfg["customers"]["activation_start"])
    activation_end = pd.Timestamp(data_cfg["customers"]["activation_end"])
    activation_seconds = int((activation_end - activation_start).total_seconds())
    activation_dates = activation_start + pd.to_timedelta(
        rng.integers(0, activation_seconds, size=n_rows), unit="s"
    )
    reference_date = pd.Timestamp(data_cfg["time_range"]["end"])

    return pd.DataFrame(
        {
            "subscriber_id": [f"SUB{i:06d}" for i in range(1, n_rows + 1)],
            "activation_date": activation_dates,
            "customer_segment": rng.choice(
                ["consumer", "business", "premium"], n_rows, p=[0.65, 0.25, 0.10]
            ),
            "plan_type": rng.choice(["prepaid", "postpaid"], n_rows, p=[0.70, 0.30]),
            "region": rng.choice(
                ["north", "south", "east", "west", "central"],
                n_rows,
                p=[0.20, 0.20, 0.15, 0.25, 0.20],
            ),
            "device_type": rng.choice(
                ["smartphone", "feature_phone"], n_rows, p=[0.75, 0.25]
            ),
            "tenure_days": (reference_date - activation_dates).days,
            "age_band": rng.choice(
                ["18_25", "26_35", "36_45", "46_60", "60_plus"],
                n_rows,
                p=[0.20, 0.30, 0.25, 0.15, 0.10],
            ),
            "preferred_language": rng.choice(
                ["english", "french", "arabic", "local"],
                n_rows,
                p=[0.20, 0.30, 0.25, 0.25],
            ),
            "account_status": rng.choice(
                ["active", "suspended", "inactive"], n_rows, p=[0.90, 0.05, 0.05]
            ),
        }
    )


def save_dataset(dataframe: pd.DataFrame, directory: Path, filename: str) -> None:
    """Save a dataframe as parquet."""
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / filename
    dataframe.to_parquet(output_path, index=False)
    print(f"Generated {filename}: {dataframe.shape} -> {output_path}")


def main() -> None:
    """Generate all synthetic datasets."""
    config = load_config()
    seed = config["data"]["seed"]

    customers = generate_customers(config)
    profiles = generate_behavior_profiles(customers, seed)
    world = build_world(customers, config, seed + 10)

    usage_events = events.generate_usage_events(config, customers, profiles, world, seed + 1)
    recharges = events.generate_recharges(config, customers, profiles, world, seed + 2)
    network_events = events.generate_network_events(config, customers, profiles, world, seed + 3)
    risk_events = events.generate_risk_events(config, customers, profiles, world, seed + 4)
    snapshot_labels = build_snapshot_labels(customers, world, config)

    save_dataset(customers, RAW_DATA_DIR, "customers.parquet")
    save_dataset(usage_events, RAW_DATA_DIR, "usage_events.parquet")
    save_dataset(recharges, RAW_DATA_DIR, "recharges.parquet")
    save_dataset(network_events, RAW_DATA_DIR, "network_events.parquet")
    save_dataset(risk_events, RAW_DATA_DIR, "risk_events.parquet")
    save_dataset(snapshot_labels, RAW_DATA_DIR, "snapshot_labels.parquet")

    # Ground truth for analysis only. Never read these when building features.
    save_dataset(world, HIDDEN_DATA_DIR, "world.parquet")
    save_dataset(profiles, HIDDEN_DATA_DIR, "profiles.parquet")

    rate = snapshot_labels["risk_label"].mean()
    print(f"Snapshots: {snapshot_labels['snapshot_date'].nunique()}")
    print(f"Rows: {len(snapshot_labels)}  positive rate: {rate:.4f}")


if __name__ == "__main__":
    main()
