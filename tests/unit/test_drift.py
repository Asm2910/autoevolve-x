import numpy as np
import pandas as pd

from monitoring.drift import (
    categorical_psi,
    feature_drift,
    numeric_psi,
    simulate_drift,
    summarize,
)


def normal(n, loc=0.0, scale=1.0, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(loc, scale, n))


def test_same_distribution_has_near_zero_psi():
    assert numeric_psi(normal(20000, seed=1), normal(20000, seed=2)) < 0.02


def test_shifted_distribution_has_large_psi():
    assert numeric_psi(normal(20000, seed=1), normal(20000, loc=1.0, seed=2)) > 0.25


def test_missing_values_count_as_a_bin():
    reference = normal(5000, seed=1)
    current = reference.copy()
    current.iloc[:2500] = np.nan  # half the values disappear
    assert numeric_psi(reference, current) > 0.25


def test_constant_and_empty_columns_do_not_crash():
    assert numeric_psi(pd.Series([1.0] * 100), pd.Series([1.0] * 100)) < 0.01
    assert numeric_psi(pd.Series([np.nan] * 10), pd.Series([1.0] * 10)) == 0.0


def test_categorical_psi_detects_a_new_mix_and_new_levels():
    reference = pd.Series(["a"] * 800 + ["b"] * 200)
    assert categorical_psi(reference, pd.Series(["a"] * 800 + ["b"] * 200)) < 0.001
    assert categorical_psi(reference, pd.Series(["a"] * 200 + ["b"] * 800)) > 0.25
    assert categorical_psi(reference, pd.Series(["a"] * 500 + ["zzz"] * 500)) > 0.25  # level never seen before


def frame(seed, shift=0.0):
    rng = np.random.default_rng(seed)
    n = 5000
    return pd.DataFrame({
        "usage__calls": rng.normal(shift, 1, n),
        "network__latency": rng.normal(0, 1, n),
        "customer__plan": rng.choice(["x", "y"], n),
    })


def test_feature_drift_table_ranks_the_shifted_feature_first():
    table = feature_drift(frame(1), frame(2, shift=1.5), ["usage__calls", "network__latency"], ["customer__plan"])
    assert table.iloc[0]["feature"] == "usage__calls"
    assert set(table["kind"]) == {"numeric", "categorical"}
    assert table.loc[table["feature"] == "network__latency", "psi"].iloc[0] < 0.05


def test_summary_decides_drift_by_share_of_major_features():
    stable = feature_drift(frame(1), frame(2), ["usage__calls", "network__latency"], ["customer__plan"])
    assert summarize(stable)["verdict"] == "stable"
    shifted = feature_drift(frame(1), frame(2, shift=1.5), ["usage__calls", "network__latency"], ["customer__plan"])
    summary = summarize(shifted)  # 1 of 3 features is major, above the 5% default
    assert summary["verdict"] == "drift" and summary["n_major"] == 1
    assert summarize(shifted, {"retrain_share": 0.5})["verdict"] == "stable"  # threshold is configurable


def test_simulate_drift_only_touches_the_chosen_group():
    original = frame(1)
    changed = simulate_drift(original, "usage", 2.0, ["customer__plan"])
    assert np.allclose(changed["usage__calls"], original["usage__calls"] * 2)
    assert changed["network__latency"].equals(original["network__latency"])
    assert changed["customer__plan"].equals(original["customer__plan"])
    assert original["usage__calls"].iloc[0] != changed["usage__calls"].iloc[0]  # input not modified in place