"""Send realistic prediction traffic to the running API, so the Grafana dashboard has something to show.

  python -m scripts.generate_traffic --count 300 --rate 10
  python -m scripts.generate_traffic --count 300 --rate 10 --shift usage:1.6   # rehearse a drifted client

Rows come from the VALIDATION snapshots (never the test snapshots). With --shift, every numeric feature of
one group is scaled, which moves the score distribution the same way real drift would.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import requests

from models.training import PROCESSED_DIR
from monitoring.drift import simulate_drift

METADATA_PATH = PROCESSED_DIR.parents[1] / "models" / "artifacts" / "model_metadata.json"


def build_payloads(frame: pd.DataFrame, required: list[str]) -> list[dict]:
    """One request body per row: the features the model needs, with missing values sent as null."""
    payloads = []
    for _, row in frame.iterrows():
        features = json.loads(row[required].to_json())
        payloads.append({"subscriber_id": str(row["subscriber_id"]), "features": features})
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--count", type=int, default=200, help="number of /predict requests")
    parser.add_argument("--rate", type=float, default=5.0, help="requests per second")
    parser.add_argument("--shift", default="", help="GROUP:FACTOR, for example usage:1.6")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    metadata = json.loads(METADATA_PATH.read_text())
    required = list(metadata["features"])
    manifest = json.loads((PROCESSED_DIR / "feature_manifest.json").read_text())
    frame = pd.read_parquet(PROCESSED_DIR / "validation.parquet").sample(args.count, replace=True, random_state=args.seed)
    if args.shift:
        group, _, factor = args.shift.partition(":")
        frame = simulate_drift(frame, group, float(factor), list(manifest["categorical"]))
        print(f"Scaling the '{group}' features by {factor} to rehearse drift")

    scores, flagged, failed = [], 0, 0
    for payload in build_payloads(frame, required):
        started = time.perf_counter()
        try:
            response = requests.post(f"{args.url}/predict", json=payload, timeout=10)
            if response.ok:
                body = response.json()
                scores.append(body["risk_probability"])
                flagged += body["risk_prediction"]
            else:
                failed += 1
        except requests.RequestException:
            failed += 1
        time.sleep(max(0.0, 1.0 / args.rate - (time.perf_counter() - started)))

    sent = len(scores)
    print(f"Sent {sent + failed} requests: {sent} scored, {failed} failed")
    if sent:
        print(f"Mean score {sum(scores) / sent:.4f}, flagged {flagged / sent:.1%}")


if __name__ == "__main__":
    main()