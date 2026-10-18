"""Run a whole mission: plan, review the data, evaluate, compare models, improve, reflect.

  python -m scripts.run_mission --goal "Find and fix where fraud is missed" --depth quick
  AUTOEVOLVE_DATASET=ieee python -m scripts.run_mission --goal "Improve fraud detection" --depth standard

Depths: quick (2 experiments), standard (4 per round, 2 rounds, model arena), thorough (6, 3 rounds).
Writes mission_report.{md,json} next to the processed data and a JSONL trace under memory/traces/.
Stop it at any time with  touch memory/STOP.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from agentkit.budget import Budget
from agentkit.tracing import Observer, TraceLog
from mission.graph import run_mission
from mission.plan import Mission
from mission.report import mission_markdown, mission_payload
from mission.services import build_services, save_report
from models.training import PROJECT_ROOT, load_config
from observability import agent_metrics as metrics
from observability.push import push_at_exit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--goal", default="Improve fraud detection and explain what changed")
    parser.add_argument("--depth", choices=["quick", "standard", "thorough"], default="standard")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None, help="experiments per improvement round")
    parser.add_argument("--workers", type=int, default=1, help="models trained at once in the arena (Ray when above 1)")
    args = parser.parse_args()
    push_at_exit("run_mission")

    platform = load_config().get("agent_platform", {})
    budget = Budget.from_config(platform, PROJECT_ROOT)
    trace = PROJECT_ROOT / platform.get("trace_dir", "memory/traces") / f"mission-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    observer = Observer(TraceLog(trace))
    services, close = build_services(budget, observer, workers=args.workers)
    try:
        mission = Mission(goal=args.goal, depth=args.depth, max_rounds=args.rounds, improve_iterations=args.iterations)
        state = run_mission(mission, services, budget=budget, observer=observer)
    finally:
        close()
    metrics.MISSION_ROUNDS.set(state.get("round", 0))
    metrics.MISSION_HALTED.set(1 if state.get("halted") else 0)
    markdown = mission_markdown(state)
    save_report(markdown, mission_payload(state))
    print("\n" + markdown)
    print(f"Budget used: {budget.snapshot()}\nTrace: {trace}")


if __name__ == "__main__":
    main()
