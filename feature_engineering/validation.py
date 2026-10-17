"""Validation for the snapshot feature dataset."""

import numpy as np
import pandas as pd

ID_COLUMNS = ["subscriber_id", "snapshot_date"]
LABEL = "risk_label"
FORBIDDEN_NAME_PARTS = ("fraud", "confirm", "hidden", "world", "latent")


def validate_feature_dataset(
    features: pd.DataFrame,
    expected_snapshots: int | None = None,
) -> None:
    """Raise ValueError listing every problem found in the feature dataset."""
    errors: list[str] = []

    missing = [c for c in (*ID_COLUMNS, LABEL) if c not in features.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feature_columns = [c for c in features.columns if c not in (*ID_COLUMNS, LABEL)]

    if features.duplicated(ID_COLUMNS).any():
        errors.append("duplicate (subscriber_id, snapshot_date) rows")

    if not features[LABEL].isin([0, 1]).all():
        errors.append("risk_label must contain only 0 and 1")

    if features[LABEL].nunique() != 2:
        errors.append("risk_label must contain both classes")

    leaking = [
        c for c in feature_columns if any(p in c.lower() for p in FORBIDDEN_NAME_PARTS)
    ]
    if leaking:
        errors.append(f"feature names look like ground truth: {leaking}")

    numeric = features[feature_columns].select_dtypes("number")
    if np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).any():
        errors.append("infinite values found in numeric features")

    flag_columns = [c for c in feature_columns if c.split("__")[1:2] == ["has_events"]]
    for column in flag_columns:
        if not features[column].isin([0, 1]).all():
            errors.append(f"{column} must be 0/1")

    if expected_snapshots is not None:
        found = features["snapshot_date"].nunique()
        if found != expected_snapshots:
            errors.append(f"expected {expected_snapshots} snapshots, found {found}")

    per_snapshot = features.groupby("snapshot_date")[LABEL].sum()
    if (per_snapshot == 0).any():
        errors.append("a snapshot has no positive labels")

    if errors:
        raise ValueError("Feature validation failed:\n- " + "\n- ".join(errors))

    print(
        f"Feature validation passed: {len(features):,} rows, "
        f"{len(feature_columns)} features, "
        f"{features['snapshot_date'].nunique()} snapshots."
    )
