"""Turn the IEEE-CIS Kaggle download into the project's processed format.

  python -m scripts.prepare_ieee --source ~/Downloads/ieee-fraud-detection.zip
  python -m scripts.prepare_ieee --source ~/Downloads/ieee-fraud-detection.zip --sample 0   # all 590k rows

Writes data/ieee/processed/{features,train,validation,test}.parquet and feature_manifest.json, plus
data/ieee/raw/{transaction,identity}.parquet for the data review. Everything under data/ieee is git-ignored.
Afterwards run any script on this dataset with  AUTOEVOLVE_DATASET=ieee  in front of it.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import datasets
from data_adapters import ieee_cis as ieee


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="the Kaggle zip, or a folder holding the train_*.csv files")
    parser.add_argument("--sample", type=int, default=300_000,
                        help="keep this many rows, spread evenly over time (0 = all). Default 300000")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed, raw = datasets.ROOT / "data" / "ieee" / "processed", datasets.ROOT / "data" / "ieee" / "raw"
    processed.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    print("Reading the Kaggle files ...")
    trans, ident = ieee.load_tables(args.source)
    trans.to_parquet(raw / "transaction.parquet", index=False)
    ident.to_parquet(raw / "identity.parquet", index=False)

    print("Building features (card history is computed on all rows, then rows are sampled) ...")
    frame, categorical = ieee.build_frame(trans, ident)
    if args.sample and len(frame) > args.sample:
        keep = np.sort(np.random.default_rng(args.seed).choice(len(frame), args.sample, replace=False))
        frame = frame.iloc[keep].reset_index(drop=True)
        print(f"  sampled {len(frame):,} rows (time order kept)")
    train, validation, test = ieee.split_by_time(frame)
    ieee.cap_levels(train, [validation, test], categorical)
    full = pd.concat([train, validation, test], ignore_index=True)

    manifest = ieee.manifest_for(full, categorical)
    full.to_parquet(processed / "features.parquet", index=False)
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        part.to_parquet(processed / f"{name}.parquet", index=False)
        print(f"  {name:<11} rows={len(part):>8,}  fraud={int(part[ieee.LABEL].sum()):>6,}  rate={part[ieee.LABEL].mean():.4f}  "
              f"days {part['snapshot_date'].min().date()} .. {part['snapshot_date'].max().date()}")
    (processed / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest['features'])} features in {len(manifest['groups'])} groups "
          f"({len(manifest['categorical'])} categorical): " + ", ".join(f"{g} {len(c)}" for g, c in manifest["groups"].items()))
    print(f"Wrote {processed}\nNext:  AUTOEVOLVE_DATASET=ieee python -m scripts.run_arena")


if __name__ == "__main__":
    main()
