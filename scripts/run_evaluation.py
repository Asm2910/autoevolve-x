"""Run the evaluation swarm on a model's validation scores.

  python -m scripts.run_evaluation                      # the configured baseline (xgboost, all features)
  python -m scripts.run_evaluation --model catboost     # any arena model

Writes data/processed/evaluation_report.{md,json} and memory/evaluation_findings.json (the ideas
the proposer agent can read with `run_agent --use-findings`). Validation data only: the sealed
test split is never loaded here.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from agentkit.budget import Budget
from agentkit.tracing import Observer, TraceLog
from agents.spec import ExperimentSpec
from arena.registry import names
from models.training import PROCESSED_DIR, PROJECT_ROOT, load_config, select_features
from swarm.evaluation import EvalContext, EvalRequest, EvaluationSwarm
from tools.experiment_runner import ExperimentRunner

FINDINGS_PATH = PROJECT_ROOT / "memory" / "evaluation_findings.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, choices=names(), help="model to evaluate (default: baseline)")
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()

    config = load_config()
    runner = ExperimentRunner(config)
    spec = runner.baseline_spec()
    if args.model:
        spec = ExperimentSpec(model=args.model, groups=spec.groups, windows=spec.windows,
                              hypothesis=f"Evaluate {args.model}")
    print(f"Scoring validation data with: {spec.key()}")
    scores, _ = runner._fit_predict(spec)

    features = select_features(runner.manifest, spec.groups, spec.windows)
    categorical = [c for c in features if c in runner.manifest["categorical"]]
    numeric = [c for c in features if c not in categorical]
    context = EvalContext(
        reference=runner.train, frame=runner.validation, y=runner.y_val, scores=scores,
        numeric=numeric, categorical=categorical, review_fraction=config["training"]["review_fraction"],
        seed=config["training"]["seed"], dates=runner.validation["snapshot_date"],
        drift_settings=config.get("drift", {}),
    )
    platform = config.get("agent_platform", {})
    trace = PROJECT_ROOT / platform.get("trace_dir", "memory/traces") / f"evaluation-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    swarm = EvaluationSwarm(context, Observer(TraceLog(trace)), Budget.from_config(platform, PROJECT_ROOT))
    report = swarm.run(EvalRequest(n_boot=args.bootstrap))
    report.save(PROCESSED_DIR, FINDINGS_PATH)

    print("\n" + report.markdown())
    print(f"Trace: {trace}\nIdeas for the proposer: {FINDINGS_PATH}")


if __name__ == "__main__":
    main()
