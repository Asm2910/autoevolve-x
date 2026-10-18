"""Hard limits for a mission, enforced in code and not by asking the model to behave.

`check()` is called before every iteration and every model call. It raises when a limit is
reached, and the agent loop turns that into a clean stop with a stated reason. A kill switch
(a file on disk) lets a human stop a run from another terminal:  touch memory/STOP
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


class BudgetExceeded(RuntimeError):
    """A mission limit was reached. The message names the limit."""


class KillSwitchTripped(RuntimeError):
    """A human asked the run to stop."""


@dataclass
class Budget:
    max_llm_calls: int = 60
    max_input_tokens: int = 400_000
    max_output_tokens: int = 60_000
    max_cost_usd: float = 2.0
    max_seconds: float = 3600.0
    max_experiments: int = 15
    # Prices are assumptions used only for the cost estimate. Set them to your plan's current rates.
    input_usd_per_million: float = 3.0
    output_usd_per_million: float = 15.0
    kill_switch_path: Path | None = None

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    experiments: int = 0
    started: float = field(default_factory=time.monotonic)

    @classmethod
    def from_config(cls, platform_config: dict, project_root: Path | None = None) -> Budget:
        cfg = platform_config or {}
        prices = cfg.get("price_per_mtok", {})
        switch = cfg.get("kill_switch_file")
        if switch and project_root is not None:
            switch = project_root / switch
        return cls(
            max_llm_calls=cfg.get("max_llm_calls", 60),
            max_input_tokens=cfg.get("max_input_tokens", 400_000),
            max_output_tokens=cfg.get("max_output_tokens", 60_000),
            max_cost_usd=cfg.get("max_cost_usd", 2.0),
            max_seconds=cfg.get("max_seconds", 3600.0),
            max_experiments=cfg.get("max_experiments", 15),
            input_usd_per_million=prices.get("input", 3.0),
            output_usd_per_million=prices.get("output", 15.0),
            kill_switch_path=Path(switch) if switch else None,
        )

    @classmethod
    def unlimited(cls) -> Budget:
        return cls(
            max_llm_calls=10**9, max_input_tokens=10**12, max_output_tokens=10**12,
            max_cost_usd=1e9, max_seconds=1e12, max_experiments=10**9,
        )

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * self.input_usd_per_million + self.output_tokens * self.output_usd_per_million
        ) / 1e6

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started

    def charge_llm(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.llm_calls += 1
        self.input_tokens += int(input_tokens)
        self.output_tokens += int(output_tokens)

    def charge_experiment(self) -> None:
        self.experiments += 1

    def check(self) -> None:
        """Raise if the run must stop. Safe to call as often as you like."""
        if self.kill_switch_path is not None and self.kill_switch_path.exists():
            raise KillSwitchTripped(f"kill switch file present: {self.kill_switch_path}")
        limits = [
            (self.elapsed_seconds >= self.max_seconds, f"time budget reached ({self.max_seconds:.0f}s)"),
            (self.llm_calls >= self.max_llm_calls, f"LLM call budget reached ({self.max_llm_calls} calls)"),
            (self.input_tokens >= self.max_input_tokens, f"input token budget reached ({self.max_input_tokens})"),
            (self.output_tokens >= self.max_output_tokens, f"output token budget reached ({self.max_output_tokens})"),
            (self.cost_usd >= self.max_cost_usd, f"cost budget reached (${self.max_cost_usd:.2f})"),
            (self.experiments >= self.max_experiments, f"experiment budget reached ({self.max_experiments})"),
        ]
        for reached, message in limits:
            if reached:
                raise BudgetExceeded(message)

    def snapshot(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "experiments": self.experiments,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }
