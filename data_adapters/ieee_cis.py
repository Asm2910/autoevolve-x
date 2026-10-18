"""IEEE-CIS Fraud Detection (Vesta / Kaggle): real e-commerce card fraud, two linked sources.

  train_transaction.csv   590,540 payments, 3.5% fraud, about 183 days; amount, card, address, e-mail
                          domain, counters (C), time deltas (D), match flags (M), Vesta features (V)
  train_identity.csv      device and browser information for about 24% of the payments

The tables join on TransactionID. Only the two *train* files are used (the Kaggle *test* files have no
labels), and this project makes its own temporal split of them: the earliest 60% of time trains, the next
20% tunes, the last 20% is the sealed test.

Feature groups follow the project's naming scheme, `<group>__<metric>`:
  transaction, card, address, email, counts, timedeltas, match, vesta, identity, device
and one extra group, `history`, built per card from earlier payments only (count so far, amount against
the card's own average, seconds since its previous payment). Those use nothing from the future, so computing
them before the split does not leak.

TransactionDT is a time offset in seconds, not a date. Dates here are that offset added to 2017-11-30,
the reference most public write-ups assume. They are only used to order and label snapshots.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_DATE = pd.Timestamp("2017-11-30")
LABEL_SOURCE = "isFraud"
LABEL = "risk_label"  # the name the rest of the project uses
MAX_LEVELS = 25  # categorical columns keep their 25 most common training values, the rest become "other"

GROUP_OF_PREFIX = {"C": "counts", "D": "timedeltas", "M": "match", "V": "vesta"}


# ---- reading -------------------------------------------------------------------------------------
def _read_csv(opener, name: str) -> pd.DataFrame:
    frame = pd.read_csv(opener)
    for column in frame.select_dtypes("float64").columns:
        frame[column] = frame[column].astype("float32")
    print(f"  read {name}: {len(frame):,} rows, {frame.shape[1]} columns")
    return frame


def load_tables(source: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read train_transaction.csv and train_identity.csv from the Kaggle zip or from a folder."""
    source = Path(source).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"{source} not found")
    if source.is_dir():
        files = {p.name: p for p in source.iterdir()}
        missing = [n for n in ("train_transaction.csv", "train_identity.csv") if n not in files]
        if missing:
            raise FileNotFoundError(f"{source} does not contain {missing}")
        return (_read_csv(files["train_transaction.csv"], "train_transaction.csv"),
                _read_csv(files["train_identity.csv"], "train_identity.csv"))
    with zipfile.ZipFile(source) as archive:
        members = {Path(n).name: n for n in archive.namelist()}
        missing = [n for n in ("train_transaction.csv", "train_identity.csv") if n not in members]
        if missing:
            raise FileNotFoundError(f"{source} does not contain {missing}")
        out = []
        for name in ("train_transaction.csv", "train_identity.csv"):
            with archive.open(members[name]) as handle:
                out.append(_read_csv(io.BytesIO(handle.read()), name))
        return out[0], out[1]


# ---- features ------------------------------------------------------------------------------------
def card_history(trans: pd.DataFrame) -> pd.DataFrame:
    """Per-card features from earlier payments only. `trans` must be sorted by TransactionDT."""
    key = pd.factorize(trans["card1"].astype(str) + "|" + trans["addr1"].astype(str))[0]
    frame = pd.DataFrame({"key": key, "amt": trans["TransactionAmt"].to_numpy(dtype="float64"),
                          "dt": trans["TransactionDT"].to_numpy(dtype="float64")}, index=trans.index)
    grouped = frame.groupby("key", sort=False)
    prior = grouped.cumcount()
    prior_sum = grouped["amt"].cumsum() - frame["amt"]
    mean_prior = (prior_sum / prior.replace(0, np.nan))
    previous = grouped["dt"].shift(1)
    return pd.DataFrame({
        "history__card_prior_count": prior.astype("float32"),
        "history__amount_vs_card_mean": (frame["amt"] / mean_prior).astype("float32"),
        "history__seconds_since_card_prev": (frame["dt"] - previous).astype("float32"),
    }, index=trans.index)


def _group_for(column: str) -> str:
    if column.startswith("id_"):
        return "identity"
    for prefix, group in GROUP_OF_PREFIX.items():
        if column[0] == prefix and column[1:].isdigit():
            return group
    raise KeyError(column)


def build_frame(trans: pd.DataFrame, ident: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One row per payment with project-style column names. Returns (frame, categorical column names)."""
    trans = trans.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    out: dict[str, np.ndarray] = {
        "TransactionID": trans["TransactionID"].to_numpy(),
        "TransactionDT": trans["TransactionDT"].to_numpy(),
        LABEL: trans[LABEL_SOURCE].to_numpy().astype("int64"),
        "snapshot_date": (REFERENCE_DATE + pd.to_timedelta(trans["TransactionDT"] // 86400, unit="D"))
        .astype("datetime64[us]").to_numpy(),
    }
    cat: list[str] = []

    def numeric(name: str, series: pd.Series) -> None:
        out[name] = series.to_numpy(dtype="float32")

    def categorical(name: str, series: pd.Series) -> None:
        out[name] = series.astype("object").where(series.notna(), "missing").astype(str).to_numpy()
        cat.append(name)

    amount = trans["TransactionAmt"]
    numeric("transaction__amount", amount)
    numeric("transaction__log_amount", np.log1p(amount))
    numeric("transaction__cents", (amount * 100) % 100)
    numeric("transaction__hour", (trans["TransactionDT"] // 3600) % 24)
    numeric("transaction__weekday", (trans["TransactionDT"] // 86400) % 7)
    categorical("transaction__product", trans["ProductCD"])
    for column in ("card1", "card2", "card3", "card5"):
        numeric(f"card__{column}", trans[column])
    categorical("card__network", trans["card4"])
    categorical("card__type", trans["card6"])
    for column in ("addr1", "addr2", "dist1", "dist2"):
        numeric(f"address__{column}", trans[column])
    categorical("email__purchaser_domain", trans["P_emaildomain"])
    categorical("email__recipient_domain", trans["R_emaildomain"])
    numeric("email__same_domain", (trans["P_emaildomain"].astype("object") == trans["R_emaildomain"].astype("object")).astype("float32"))
    for column in trans.columns:
        if column[0] in "CDV" and column[1:].isdigit():
            numeric(f"{_group_for(column)}__{column.lower()}", trans[column])
    for column in (c for c in trans.columns if c[0] == "M" and c[1:].isdigit()):
        categorical(f"match__{column.lower()}", trans[column])

    ident = ident.drop_duplicates("TransactionID").set_index("TransactionID").reindex(trans["TransactionID"])
    numeric("identity__has_identity", ident["id_01"].notna())
    for column in ident.columns:
        if not column.startswith("id_"):
            continue
        name = f"identity__{column}"
        if pd.api.types.is_numeric_dtype(ident[column]):
            numeric(name, ident[column])
        else:
            categorical(name, ident[column])
    categorical("device__type", ident["DeviceType"])
    categorical("device__info", ident["DeviceInfo"].astype("object").str.lower().str.split().str[0])

    history = card_history(trans)
    for column in history.columns:
        out[column] = history[column].to_numpy()
    return pd.DataFrame(out), cat


def cap_levels(train: pd.DataFrame, others: list[pd.DataFrame], categorical: list[str], max_levels: int = MAX_LEVELS) -> None:
    """Keep each column's most common training values; everything else becomes 'other'. Uses train rows only."""
    for column in categorical:
        keep = set(train[column].value_counts().head(max_levels).index)
        for frame in (train, *others):
            frame[column] = frame[column].where(frame[column].isin(keep), "other")


def split_by_time(frame: pd.DataFrame, train_share: float = 0.6, validation_share: float = 0.2):
    """Earliest rows train, next validate, latest are the sealed test. Rows with equal time stay together."""
    times = frame["TransactionDT"].to_numpy()
    n = len(frame)
    cut_train = times[int(n * train_share)]
    cut_validation = times[int(n * (train_share + validation_share))]
    train = frame[times < cut_train]
    validation = frame[(times >= cut_train) & (times < cut_validation)]
    test = frame[times >= cut_validation]
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def manifest_for(frame: pd.DataFrame, categorical: list[str]) -> dict:
    features = [c for c in frame.columns if "__" in c]
    groups: dict[str, list[str]] = {}
    for column in features:
        groups.setdefault(column.split("__")[0], []).append(column)
    return {"id_columns": ["TransactionID", "snapshot_date"], "label": LABEL, "features": features,
            "categorical": [c for c in categorical if c in features], "groups": groups}
