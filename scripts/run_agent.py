"""Run the experiment agent.

  python -m scripts.run_agent --iterations 4

Each iteration the agent proposes one experiment, runs it on the training and
validation snapshots, and adopts it only if validation PR-AUC clearly improves.
The test snapshots are never touched here.
"""

from __future__ import annotations

import argparse

from agents.graph import build_agent, load_champion
from agents.proposers import RuleProposer
from memory.embedding import get_embedder
from memory.store import MemoryStore
from memory.vector import VectorIndex
from models.training import PROCESSED_DIR, load_config
from scripts.train_models import file_sha256
from tools.experiment_runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=None, help="experiments to run (default: from config)")
    parser.add_argument("--embedder", choices=["minilm", "hashing", "none"], default="minilm")
    args = parser.parse_args()

    config = load_config()
    max_iterations = args.iterations or int(config["agent"]["max_iterations"])
    dataset_sha256 = file_sha256(PROCESSED_DIR / "features.parquet")

    vectors = None if args.embedder == "none" else VectorIndex(get_embedder(args.embedder))
    store = MemoryStore(vectors=vectors)
    runner = ExperimentRunner(config)
    champion = load_champion(store, runner, dataset_sha256)
    print(f"Champion at start: {champion.key()}")
    runner.set_champion(champion)

    agent = build_agent(runner, store, RuleProposer(config["models"]["xgboost"]), config, dataset_sha256)
    state = agent.invoke(
        {
            "iteration": 0,
            "max_iterations": max_iterations,
            "retries": 0,
            "champion": {**champion.identity(), "hypothesis": champion.hypothesis},
            "history": [],
        },
        config={"recursion_limit": 200},
    )

    print("\nSummary")
    for i, entry in enumerate(state["history"], start=1):
        print(f"  {i}. [{entry['verdict']}] {entry['hypothesis']}")
    if state.get("stop_reason"):
        print(f"Stopped early: {state['stop_reason']}")
    adopted = [e for e in state["history"] if e["verdict"] == "adopt"]
    print(f"\nAdopted {len(adopted)} of {len(state['history'])}. Champion now: "
          f"{runner.champion.key()}")
    print("The test snapshots were not used. Confirm a champion on test only once, at the end.")


if __name__ == "__main__":
    main()