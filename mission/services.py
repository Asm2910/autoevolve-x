"""The real services behind the mission graph: they call the swarms, the arena, the judge and the agent.

Each service returns plain dicts so the graph, the report and the tests need no pandas objects.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd
from sklearn.metrics import average_precision_score

import datasets
from agentkit.budget import Budget
from agentkit.data_access import load_split
from agentkit.tracing import Observer
from agents.graph import build_agent, load_champion
from agents.proposers import RuleProposer
from arena.executor import make_executor
from arena.runner import ArenaData, build_tasks, run_arena
from judge.utility import Candidate, UtilityConfig, judge
from memory.store import MemoryStore
from mission.graph import Services
from models.training import LABEL, PROCESSED_DIR, PROJECT_ROOT, load_config, select_features
from scripts.train_models import file_sha256
from swarm.data import DataContext, DataRequest, DataSwarm, ieee_specs, telecom_specs
from swarm.evaluation import EvalContext, EvalRequest, EvaluationReport, EvaluationSwarm
from tools.experiment_runner import ExperimentRunner


def summarize(report: EvaluationReport) -> dict:
    return {"verdict": report.verdict, "failed": report.failed, "hypotheses": report.hypotheses(),
            "summaries": report.summaries, "findings": [f.model_dump() for f in report.findings]}


def build_services(budget: Budget, observer: Observer, *, proposer=None, workers: int = 1, bootstrap: int = 200,
                   log: Callable[[str], None] = print) -> tuple[Services, Callable[[], None]]:
    """Wire the real components. Returns the services and a function that closes the memory store."""
    config = load_config()
    seed = config["training"]["seed"]
    review_fraction = config["training"]["review_fraction"]
    runner = ExperimentRunner(config)
    store = MemoryStore()
    dataset_sha256 = file_sha256(PROCESSED_DIR / "features.parquet")
    manifest = runner.manifest
    ieee = datasets.active() == "ieee"

    def data_review() -> dict:
        import yaml

        specs = ieee_specs() if ieee else telecom_specs()
        raw = datasets.raw_dir()
        tables = {name: pd.read_parquet(raw / f"{name}.parquet") for name in specs}
        window = None
        if not ieee:
            cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "data.yaml").read_text())["data"]["time_range"]
            window = (pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"]))
        train = load_split("train", PROCESSED_DIR)
        context = DataContext(
            tables=tables, specs=specs, time_window=window,
            parent_table="transaction" if ieee else "customers", label_table=None if ieee else "snapshot_labels",
            entity_col="TransactionID" if ieee else "subscriber_id", features=train[manifest["features"]],
            y=train[LABEL].to_numpy(), categorical=list(manifest["categorical"]), seed=seed)
        return summarize(DataSwarm(context, observer, budget).run(DataRequest()))

    def evaluate() -> dict:
        champion = load_champion(store, runner, dataset_sha256)  # the model as the memory remembers it
        scores, _ = runner._fit_predict(champion)
        features = select_features(manifest, champion.groups, champion.windows)
        categorical = [c for c in features if c in manifest["categorical"]]
        context = EvalContext(
            reference=runner.train, frame=runner.validation, y=runner.y_val, scores=scores,
            numeric=[c for c in features if c not in categorical], categorical=categorical,
            review_fraction=review_fraction, seed=seed, dates=runner.validation["snapshot_date"],
            drift_settings=config.get("drift", {}))
        out = summarize(EvaluationSwarm(context, observer, budget).run(EvalRequest(n_boot=bootstrap)))
        out["pr_auc"] = float(average_precision_score(runner.y_val, scores))
        return out

    def arena() -> dict:
        models = config.get("arena", {}).get("models") or config["training"]["candidates"]
        features = select_features(manifest, config["features"]["groups"], config["features"]["windows"])
        categorical = [c for c in features if c in manifest["categorical"]]
        numeric = [c for c in features if c not in categorical]
        tasks = build_tasks(config, models, numeric, categorical)
        data = ArenaData(runner.train[features], runner.train[LABEL].to_numpy(), runner.validation[features], runner.y_val)
        executor = make_executor(workers, config.get("arena", {}).get("task_timeout_seconds"), log)
        try:
            results = run_arena(tasks, data, review_fraction, executor, log)
        finally:
            executor.close()
        good = [r for r in results if r.ok]
        verdict = judge([Candidate(r.model, r.val_probs, r.train_seconds, r.size_mb) for r in good], runner.y_val,
                        UtilityConfig.from_config(config), review_fraction, n_boot=bootstrap, seed=seed)
        (PROCESSED_DIR / "mission_judge_report.md").write_text(verdict.markdown())
        return {"winner": verdict.winner, "rationale": verdict.rationale, "rows": verdict.rows,
                "failed": [r.model for r in results if not r.ok]}

    def improve(iterations: int, focus: list[str]) -> dict:
        champion = load_champion(store, runner, dataset_sha256)
        runner.set_champion(champion)
        start = float(average_precision_score(runner.y_val, runner.champion_probs))
        agent = build_agent(runner, store, proposer or RuleProposer(config["models"]["xgboost"]), config,
                            dataset_sha256, log=log, budget=budget, observer=observer, findings=focus)
        state = agent.invoke({"iteration": 0, "max_iterations": iterations, "retries": 0,
                              "champion": {**champion.identity(), "hypothesis": champion.hypothesis}, "history": []},
                             config={"recursion_limit": 200})
        end = float(average_precision_score(runner.y_val, runner.champion_probs))
        return {"history": state["history"], "champion": runner.champion.key(), "start_pr_auc": start,
                "end_pr_auc": end, "stop_reason": state.get("stop_reason", "")}

    def remember(text: str, tags: str) -> None:
        store.add_lesson(text, tags=tags)

    return Services(data_review, evaluate, arena, improve, remember), store.close


def save_report(markdown: str, payload: dict, directory=PROCESSED_DIR) -> None:
    (directory / "mission_report.md").write_text(markdown)
    (directory / "mission_report.json").write_text(json.dumps(payload, indent=2, default=str))

