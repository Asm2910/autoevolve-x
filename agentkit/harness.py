"""The base class every agent is built on.

An agent here is not a prompt. It is a unit with a written responsibility, a typed input and
output, a time limit, a list of tools it may use, and a record of every run. The harness
enforces all of that, so individual agents contain only their own logic.

  run(payload)
    1. check the budget and the kill switch
    2. validate the input against the contract
    3. execute (with a timeout, retrying transient failures)
    4. validate the output against the contract
    5. record the step in the trace log and the metrics

Python cannot kill a thread, so the timeout stops *waiting* for a stuck step and reports it.
Work that may run long (model training) belongs in Ray tasks, which can be cancelled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from agentkit.budget import Budget
from agentkit.tracing import Observer


class TransientAgentError(RuntimeError):
    """A failure worth retrying (network blip, rate limit). Everything else fails immediately."""


class ToolNotAllowed(PermissionError):
    """The agent tried to use a tool its contract does not list."""


class AgentTimeout(TimeoutError):
    """The step took longer than its contract allows."""


@dataclass(frozen=True)
class AgentContract:
    name: str
    responsibility: str  # one sentence: why this agent exists
    kind: str  # "llm" (judgement) or "deterministic" (ordinary tested code)
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    timeout_seconds: float = 120.0
    max_retries: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"llm", "deterministic"}:
            raise ValueError(f"kind must be 'llm' or 'deterministic', got {self.kind!r}")


RETRYABLE = (TransientAgentError, ConnectionError, AgentTimeout)


class BaseAgent(ABC):
    contract: AgentContract

    def __init__(self, observer: Observer | None = None, budget: Budget | None = None, tools: dict | None = None) -> None:
        self.observer = observer or Observer()
        self.budget = budget or Budget.unlimited()
        self._tools = tools or {}

    @abstractmethod
    def execute(self, data: BaseModel) -> BaseModel | dict:
        """The agent's own logic. Receives validated input; returns output for the contract."""

    def use_tool(self, name: str, *args, **kwargs):
        if name not in self.contract.allowed_tools:
            raise ToolNotAllowed(f"{self.contract.name} may not use tool {name!r}")
        with self.observer.step(f"{self.contract.name}.tool.{name}"):
            return self._tools[name](*args, **kwargs)

    def run(self, payload: BaseModel | dict) -> BaseModel:
        contract = self.contract
        self.budget.check()
        data = payload if isinstance(payload, contract.input_model) else contract.input_model.model_validate(payload)
        attempts = contract.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                with self.observer.step(contract.name, kind=contract.kind, attempt=attempt) as outcome:
                    try:
                        result = self._execute_with_timeout(data)
                        output = (
                            result
                            if isinstance(result, contract.output_model)
                            else contract.output_model.model_validate(result)
                        )
                    except ValidationError as exc:  # the agent produced output outside its contract
                        raise ValueError(
                            f"{contract.name} broke its output contract: {exc.error_count()} problem(s)"
                        ) from exc
                    outcome["output_type"] = contract.output_model.__name__
                    return output
            except RETRYABLE:
                if attempt == attempts:
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    def _execute_with_timeout(self, data: BaseModel):
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            return pool.submit(self.execute, data).result(timeout=self.contract.timeout_seconds)
        except FutureTimeout:
            raise AgentTimeout(f"{self.contract.name} exceeded {self.contract.timeout_seconds:.0f}s") from None
        finally:
            pool.shutdown(wait=False, cancel_futures=True)  # do not wait for a stuck step
