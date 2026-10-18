"""The IEEE-CIS adapter, tested on a small table with the same layout as the Kaggle files."""

import zipfile

import numpy as np
import pandas as pd
import pytest

import datasets
from data_adapters import ieee_cis as ieee

N = 800


@pytest.fixture(scope="module")
def tables():
    rng = np.random.default_rng(0)
    trans = pd.DataFrame({
        "TransactionID": np.arange(2_987_000, 2_987_000 + N),
        "isFraud": (rng.random(N) < 0.05).astype(int),
        "TransactionDT": np.sort(rng.integers(86_400, 86_400 * 180, N)),
        "TransactionAmt": rng.gamma(2.0, 50.0, N).round(2),
        "ProductCD": rng.choice(["W", "C", "H"], N),
        "card1": rng.integers(1000, 1020, N), "card2": rng.choice([100.0, 200.0, np.nan], N),
        "card3": 150.0, "card4": rng.choice(["visa", "mastercard", None], N), "card5": 226.0,
        "card6": rng.choice(["debit", "credit"], N),
        "addr1": rng.choice([300.0, 400.0, np.nan], N), "addr2": 87.0,
        "dist1": rng.choice([1.0, 5.0, np.nan], N), "dist2": np.nan,
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], N),
        "R_emaildomain": rng.choice(["gmail.com", None], N),
        "C1": rng.integers(1, 5, N).astype(float), "C2": rng.integers(1, 5, N).astype(float),
        "D1": rng.choice([0.0, 5.0, np.nan], N), "D2": rng.choice([1.0, np.nan], N),
        "M1": rng.choice(["T", "F", None], N), "M2": rng.choice(["T", "F"], N),
        "V1": rng.choice([0.0, 1.0, np.nan], N), "V2": rng.normal(size=N), "V3": rng.normal(size=N),
    })
    ident = pd.DataFrame({
        "TransactionID": trans["TransactionID"].iloc[::4].to_numpy(),
        "id_01": -5.0, "id_02": rng.normal(size=N // 4), "id_12": rng.choice(["NotFound", "Found"], N // 4),
        "id_30": "Windows 10", "id_31": "chrome 63.0",
        "DeviceType": rng.choice(["desktop", "mobile"], N // 4), "DeviceInfo": rng.choice(["Windows", "SM-G960F Build/X", None], N // 4),
    })
    return trans, ident


@pytest.fixture(scope="module")
def built(tables):
    return ieee.build_frame(*tables)


def test_columns_follow_the_project_naming_scheme(built):
    frame, categorical = built
    features = [c for c in frame.columns if "__" in c]
    assert all(len(c.split("__")) == 2 for c in features)  # no window token, so window filters never drop them
    groups = {c.split("__")[0] for c in features}
    assert groups == {"transaction", "card", "address", "email", "counts", "timedeltas", "match", "vesta",
                      "identity", "device", "history"}
    assert set(categorical) <= set(features) and "match__m1" in categorical and "transaction__product" in categorical
    assert not any(c in frame.columns for c in ("isFraud",)) and ieee.LABEL in frame.columns
    assert frame["snapshot_date"].dtype == "datetime64[us]"


def test_rows_are_sorted_in_time_and_identity_joins_left(built, tables):
    frame, _ = built
    assert frame["TransactionDT"].is_monotonic_increasing and len(frame) == N
    assert frame["identity__has_identity"].mean() == pytest.approx(0.25, abs=0.01)
    assert frame["identity__id_02"].isna().mean() > 0.7  # most payments have no identity row


def test_card_history_uses_only_earlier_payments():
    trans = pd.DataFrame({"TransactionDT": [10, 20, 30, 40], "TransactionAmt": [10.0, 30.0, 50.0, 20.0],
                          "card1": [1, 1, 1, 2], "addr1": [5, 5, 5, 5]})
    h = ieee.card_history(trans)
    assert h["history__card_prior_count"].tolist() == [0, 1, 2, 0]
    assert np.isnan(h["history__amount_vs_card_mean"].iloc[0])
    assert h["history__amount_vs_card_mean"].iloc[1] == pytest.approx(3.0)   # 30 against the earlier mean of 10
    assert h["history__amount_vs_card_mean"].iloc[2] == pytest.approx(2.5)   # 50 against mean(10, 30) = 20
    assert h["history__seconds_since_card_prev"].tolist()[1:3] == [10.0, 10.0]
    # changing a later payment must not change an earlier row
    changed = trans.copy()
    changed.loc[3, "TransactionAmt"] = 9999.0
    changed.loc[2, "TransactionAmt"] = 9999.0
    h2 = ieee.card_history(changed)
    assert h2["history__amount_vs_card_mean"].iloc[1] == h["history__amount_vs_card_mean"].iloc[1]


def test_split_is_chronological_with_no_overlap(built):
    frame, _ = built
    train, validation, test = ieee.split_by_time(frame)
    assert len(train) + len(validation) + len(test) == len(frame)
    assert train["TransactionDT"].max() < validation["TransactionDT"].min()
    assert validation["TransactionDT"].max() < test["TransactionDT"].min()
    assert 0.55 < len(train) / len(frame) < 0.65


def test_rare_levels_are_chosen_from_training_rows_only():
    train = pd.DataFrame({"c": ["a"] * 10 + ["b"] * 5 + ["c"]})
    later = pd.DataFrame({"c": ["z"] * 50 + ["a"]})
    ieee.cap_levels(train, [later], ["c"], max_levels=2)
    assert set(train["c"]) == {"a", "b", "other"}
    assert set(later["c"]) == {"other", "a"}  # 'z' is common later but unseen in training


def test_manifest_lists_features_groups_and_categoricals(built):
    frame, categorical = built
    manifest = ieee.manifest_for(frame, categorical)
    assert manifest["label"] == "risk_label" and "TransactionID" in manifest["id_columns"]
    assert sum(len(v) for v in manifest["groups"].values()) == len(manifest["features"])
    assert set(manifest["categorical"]) == set(categorical)


def test_tables_load_from_a_zip_and_from_a_folder(tables, tmp_path):
    trans, ident = tables
    archive = tmp_path / "ieee.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("train_transaction.csv", trans.to_csv(index=False))
        z.writestr("train_identity.csv", ident.to_csv(index=False))
        z.writestr("test_transaction.csv", "TransactionID\n1\n")  # ignored: unlabelled
    t, i = ieee.load_tables(archive)
    assert len(t) == N and len(i) == len(ident) and t["TransactionAmt"].dtype == "float32"
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "train_transaction.csv").write_text(trans.to_csv(index=False))
    (folder / "train_identity.csv").write_text(ident.to_csv(index=False))
    assert len(ieee.load_tables(folder)[0]) == N
    with pytest.raises(FileNotFoundError):
        ieee.load_tables(tmp_path / "nope.zip")
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("other.csv", "a\n1\n")
    with pytest.raises(FileNotFoundError):
        ieee.load_tables(bad)


def test_dataset_switch_changes_every_path_together(monkeypatch):
    monkeypatch.delenv("AUTOEVOLVE_DATASET", raising=False)
    assert datasets.active() == "telecom" and datasets.processed_dir().name == "processed"
    assert datasets.config_path().name == "models.yaml" and datasets.memory_suffix() == ""
    monkeypatch.setenv("AUTOEVOLVE_DATASET", "ieee")
    assert datasets.processed_dir().parts[-2:] == ("ieee", "processed")
    assert datasets.config_path().name == "ieee.yaml" and datasets.artifact_dir().name == "artifacts_ieee"
    assert datasets.memory_suffix() == "_ieee"
    monkeypatch.setenv("AUTOEVOLVE_DATASET", "mystery")
    with pytest.raises(SystemExit):
        datasets.active()


def test_the_ieee_config_is_complete_and_matches_the_adapter_groups():
    import yaml

    cfg = yaml.safe_load((datasets.ROOT / "configs" / "ieee.yaml").read_text())
    from arena.registry import names

    assert set(cfg["training"]["candidates"]) <= set(names()) and set(cfg["arena"]["models"]) <= set(names())
    for model in cfg["training"]["candidates"] + cfg["arena"]["models"]:
        assert model in cfg["models"]
    assert set(cfg["features"]["groups"]) == {"transaction", "card", "address", "email", "counts", "timedeltas",
                                              "match", "vesta", "identity", "device", "history"}
