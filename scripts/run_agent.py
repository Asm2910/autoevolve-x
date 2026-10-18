"""Run the experiment agent.

  python -m scripts.run_agent --iterations 4                  # rule-based proposer, no API needed
  python -m scripts.run_agent --iterations 4 --proposer llm   # Claude proposes (needs ANTHROPIC_API_KEY)

Each iteration the agent proposes one experiment, runs it on the training and
validation snapshots, and adopts it only if validation PR-AUC clearly improves.
The test snapshots are never touched here.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from agentkit.budget import Budget
from agentkit.tracing import Observer, TraceLog
from agents.graph import build_agent, load_champion
from agents.proposers import RuleProposer
from memory.embedding import get_embedder
from memory.store import MemoryStore
from memory.vector import VectorIndex
from models.training import PROCESSED_DIR, PROJECT_ROOT, load_config
from observability.push import push_at_exit
from scripts.train_models import file_sha256
from tools.experiment_runner import ExperimentRunner


def load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment (never overriding real variables)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def make_proposer(kind: str, config: dict, budget: Budget, observer: Observer):
    rules = RuleProposer(config["models"]["xgboost"])
    if kind == "rules":
        return rules

    load_env_file()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Put ANTHROPIC_API_KEY=... in the .env file "
            "(which is git-ignored) or export it, or run with --proposer rules."
        )
    from langchain_anthropic import ChatAnthropic

    from agents.llm_proposer import FallbackProposer, LLMProposer

    agent_cfg = config["agent"]
    # no `temperature`: current Claude models only accept their default
    llm = ChatAnthropic(model=agent_cfg["llm_model"], max_tokens=1000, timeout=60, max_retries=2)
    primary = LLMProposer(
        llm,
        config["models"]["xgboost"],
        list(config["features"]["groups"]),
        list(config["features"]["windows"]),
        float(agent_cfg["min_gain"]),
        budget=budget,
        observer=observer,
        max_context_tokens=int(config.get("agent_platform", {}).get("max_context_tokens", 6000)),
    )
    return FallbackProposer(primary, rules)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=None, help="experiments to run (default: from config)")
    parser.add_argument("--proposer", choices=["rules", "llm"], default="rules")
    parser.add_argument("--use-findings", action="store_true",
                        help="give the proposer the evaluation and data swarms' ideas (run scripts.run_evaluation / run_data_review first)")
    parser.add_argument("--embedder", choices=["minilm", "hashing", "none"], default="minilm")
    args = parser.parse_args()
    push_at_exit("run_agent")

    config = load_config()
    max_iterations = args.iterations or int(config["agent"]["max_iterations"])
    dataset_sha256 = file_sha256(PROCESSED_DIR / "features.parquet")
    platform = config.get("agent_platform", {})
    budget = Budget.from_config(platform, PROJECT_ROOT)
    trace_path = PROJECT_ROOT / platform.get("trace_dir", "memory/traces") / f"agent-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    observer = Observer(TraceLog(trace_path))
    proposer = make_proposer(args.proposer, config, budget, observer)

    vectors = None if args.embedder == "none" else VectorIndex(get_embedder(args.embedder))
    store = MemoryStore(vectors=vectors)
    runner = ExperimentRunner(config)
    champion = load_champion(store, runner, dataset_sha256)
    print(f"Proposer: {args.proposer}")
    print(f"Champion at start: {champion.key()}")
    runner.set_champion(champion)

    findings = []
    if args.use_findings:
        for name in ("evaluation_findings.json", "data_findings.json"):
            path = PROJECT_ROOT / "memory" / name
            if path.exists():
                findings += json.loads(path.read_text())
    if findings:
        print(f"Evaluation findings given to the proposer: {len(findings)}")
    agent = build_agent(runner, store, proposer, config, dataset_sha256, budget=budget, observer=observer, findings=findings)
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
    print(f"\nAdopted {len(adopted)} of {len(state['history'])}. Champion now: {runner.champion.key()}")
    print(f"Budget used: {budget.snapshot()}")
    print(f"Trace: {trace_path}")
    print("To stop a running agent from another terminal:  touch memory/STOP")
    print("The test snapshots were not used. Confirm a champion on test only once, at the end.")


if __name__ == "__main__":
    main()