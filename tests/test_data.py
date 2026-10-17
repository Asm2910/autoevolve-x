import json
from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path("data/processed")
FEATURES = PROCESSED / "features.parquet"
MANIFEST = PROCESSED / "feature_manifest.json"


@pytest.fixture(scope="module")
def features():
    if not FEATURES.exists():
        pytest.skip("features.parquet not found - run the pipeline first")
    return pd.read_parquet(FEATURES)


@pytest.fixture(scope="module")
def manifest():
    if not MANIFEST.exists():
        pytest.skip("feature_manifest.json not found - run the pipeline first")
    return json.loads(MANIFEST.read_text())


def test_one_row_per_subscriber_per_snapshot(features):
    assert not features.duplicated(["subscriber_id", "snapshot_date"]).any()


def test_label_is_binary_and_has_no_nulls(features):
    assert features["risk_label"].isin([0, 1]).all()


def test_positive_rate_is_realistic(features):
    assert 0.003 <= features["risk_label"].mean() <= 0.03


def test_every_snapshot_has_positives(features):
    assert (features.groupby("snapshot_date")["risk_label"].sum() > 0).all()


def test_manifest_matches_feature_columns(features, manifest):
    expected = [c for c in features.columns if c not in ("subscriber_id", "snapshot_date", "risk_label")]
    assert manifest["features"] == expected


def test_no_feature_looks_like_ground_truth(features):
    banned = ("fraud", "confirm", "hidden", "world", "latent")
    assert not [c for c in features.columns if any(word in c.lower() for word in banned)]


def test_splits_are_in_time_order():
    paths = [PROCESSED / f"{name}.parquet" for name in ("train", "validation", "test")]
    if not all(path.exists() for path in paths):
        pytest.skip("split files not found - run the pipeline first")
    train, validation, test = (pd.read_parquet(path, columns=["snapshot_date"]) for path in paths)
    assert train["snapshot_date"].max() < validation["snapshot_date"].min()
    assert validation["snapshot_date"].max() < test["snapshot_date"].min()
