"""Who chooses the next experiment.

A proposer sees the current champion and what has already been tried, and returns an
ExperimentSpec (or None when it has nothing left to suggest). RuleProposer walks a
fixed list of sensible variations with no language model. An LLM-based proposer will
implement the same interface.
"""

from __future__ import annotations

from typing import Protocol

from agents.spec import PARAM_BOUNDS, ExperimentSpec


class Proposer(Protocol):
    def propose(self, champion: ExperimentSpec, context: dict) -> ExperimentSpec | None: ...


def _clip(model: str, name: str, value: float) -> float:
    low, high, is_int = PARAM_BOUNDS[model][name]
    value = min(max(value, low), high)
    return float(round(value)) if is_int else float(value)


class RuleProposer:
    def __init__(self, base_params: dict) -> None:
        self.base_params = base_params

    def candidates(self, champion: ExperimentSpec) -> list[ExperimentSpec]:
        if champion.model != "xgboost":
            return []
        current = champion.resolved_params(self.base_params)
        windows = sorted(champion.windows)

        def variant(hypothesis: str, **changes: float) -> ExperimentSpec:
            params = dict(champion.params)
            for name, value in changes.items():
                params[name] = _clip(champion.model, name, value)
            return champion.model_copy(update={"params": params, "hypothesis": hypothesis})

        depth, lr, trees = current["max_depth"], current["learning_rate"], current["n_estimators"]
        found = [
            variant("Deeper trees capture feature interactions", max_depth=depth + 1),
            variant("Shallower trees may generalise better across months", max_depth=depth - 1),
            variant("Slower learning with more trees is often more accurate",
                    learning_rate=lr / 2, n_estimators=trees * 2),
            variant("Sampling more columns per tree may help the strong usage features",
                    colsample_bytree=0.8),
            variant("Sampling fewer columns per tree reduces dependence on one feature group",
                    colsample_bytree=0.3),
            variant("A higher minimum child weight regularises rare-event splits", min_child_weight=10),
            variant("A lower minimum child weight lets the model use rare patterns", min_child_weight=2),
        ]
        if len(windows) > 1:
            for dropped in windows:
                kept = [w for w in windows if w != dropped]
                found.append(
                    champion.model_copy(
                        update={"windows": kept, "hypothesis": f"The {dropped}-day window may add noise"}
                    )
                )
        return found

    def propose(self, champion: ExperimentSpec, context: dict) -> ExperimentSpec | None:
        tried = set(context.get("tried_keys", []))
        for spec in self.candidates(champion):
            if spec.key() not in tried and spec.key() != champion.key():
                return spec
        return None