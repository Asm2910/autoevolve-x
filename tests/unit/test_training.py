import numpy as np

from models.training import select_features, sweep_thresholds, top_fraction_stats

MANIFEST = {
    "features": [
        "customer__tenure_days",
        "usage__total_calls__7d",
        "usage__total_calls__30d",
        "usage__total_calls__90d",
        "usage__total_calls_trend__7d_vs_90d",
        "usage__total_calls_trend__30d_vs_90d",
        "risk__total_risk_events__30d",
    ]
}


def test_select_features_by_group():
    selected = select_features(MANIFEST, ["customer", "risk"], [7, 30, 90])
    assert selected == ["customer__tenure_days", "risk__total_risk_events__30d"]


def test_select_features_by_window():
    selected = select_features(MANIFEST, ["usage"], [30])
    assert selected == ["usage__total_calls__30d"]


def test_trend_feature_needs_both_of_its_windows():
    assert "usage__total_calls_trend__7d_vs_90d" not in select_features(MANIFEST, ["usage"], [7, 30])
    assert "usage__total_calls_trend__7d_vs_90d" in select_features(MANIFEST, ["usage"], [7, 90])


def test_threshold_sweep_finds_the_separating_threshold():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.2, 0.3, 0.8, 0.9])
    best, table = sweep_thresholds(y, p, np.round(np.arange(0.05, 0.96, 0.05), 2))
    assert 0.3 < best <= 0.8
    assert table["f1"].max() == 1.0


def test_top_fraction_precision_and_recall():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    precision, recall = top_fraction_stats(y, p, 0.2)  # reviews the top 2 rows
    assert precision == 0.5
    assert recall == 0.5
