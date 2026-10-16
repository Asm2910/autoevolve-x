from pathlib import Path

import pandas as pd
import pytest

FEATURES = Path("data/processed/features.parquet")


@pytest.fixture(scope="module")
def features():
    if not FEATURES.exists():
        pytest.skip("features.parquet not found")
    return pd.read_parquet(FEATURES)


def test_one_row_per_subscriber(features):
    assert features["subscriber_id"].is_unique


def test_label_is_binary_and_has_no_nulls(features):
    assert features["risk_label"].isin([0, 1]).all()


def test_positive_rate_is_about_five_percent(features):
    assert 0.04 <= features["risk_label"].mean() <= 0.06


def test_expected_feature_count(features):
    # 88 columns = 86 features + subscriber_id + risk_label
    assert features.shape[1] == 88
