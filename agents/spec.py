"""The only kind of experiment the agent is allowed to run.

Whoever proposes an experiment (a rule, or later a language model) must produce an
ExperimentSpec. It names a model, feature groups, windows and a few hyperparameter
overrides. Anything outside the allowed lists or bounds is rejected before any
training happens, so a proposer can only choose among safe, well-defined actions.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field, field_validator

from arena.registry import names, param_bounds

# name -> (minimum, maximum, integer?) for every registered model
PARAM_BOUNDS = param_bounds()


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    groups: list[str] = Field(min_length=1)
    windows: list[int] = Field(min_length=1)
    params: dict[str, float] = Field(default_factory=dict)
    hypothesis: str = ""

    @field_validator("model")
    @classmethod
    def _known_model(cls, value: str) -> str:
        if value not in names():
            raise ValueError(f"unknown model {value!r}; choose from {names()}")
        return value

    def problems(self, allowed_groups: list[str], allowed_windows: list[int]) -> list[str]:
        """Everything wrong with this spec; an empty list means it is safe to run."""
        errors = []
        unknown_groups = sorted(set(self.groups) - set(allowed_groups))
        if unknown_groups:
            errors.append(f"unknown feature groups: {unknown_groups}")
        unknown_windows = sorted(set(self.windows) - set(allowed_windows))
        if unknown_windows:
            errors.append(f"unknown windows: {unknown_windows}")
        bounds = PARAM_BOUNDS[self.model]
        for name, value in self.params.items():
            if name not in bounds:
                errors.append(f"parameter '{name}' is not allowed for {self.model}")
                continue
            low, high, _ = bounds[name]
            if not low <= value <= high:
                errors.append(f"parameter '{name}'={value} outside [{low}, {high}]")
        return errors

    def resolved_params(self, base: dict) -> dict:
        """Base config values with this spec's overrides applied (ints stay ints)."""
        merged = dict(base)
        for name, value in self.params.items():
            is_int = PARAM_BOUNDS[self.model][name][2]
            merged[name] = round(value) if is_int else float(value)
        return merged

    def key(self) -> str:
        """Identity of the experiment (ignores the free-text hypothesis)."""
        params = json.dumps({k: round(float(v), 6) for k, v in sorted(self.params.items())})
        groups = ",".join(sorted(self.groups))
        windows = ",".join(str(w) for w in sorted(self.windows))
        return f"{self.model}|{groups}|{windows}|{params}"

    def identity(self) -> dict:
        """The part of the spec that defines the experiment, for storage in memory."""
        return {
            "model": self.model,
            "groups": sorted(self.groups),
            "windows": sorted(self.windows),
            "params": {k: round(float(v), 6) for k, v in sorted(self.params.items())},
        }