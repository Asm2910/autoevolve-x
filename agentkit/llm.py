"""Language-model plumbing: token accounting, and doubles that let a mission run without an API.

  UsageTracker   a LangChain callback that reads token counts from each model response
  RecordingLLM   wraps a real model and writes every request and reply to a cassette file
  ReplayLLM      answers from a cassette, so a recorded mission can be re-run in CI with no key
  ScriptedLLM    answers from a fixed list, for tests that need exact control

All of them expose the small surface the proposer uses: `with_structured_output(schema, method)`
returning an object with `invoke(messages, config=None)`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

try:  # langchain is installed with the agent; the doubles below work without it
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # pragma: no cover
    BaseCallbackHandler = object  # type: ignore[assignment,misc]


class UsageTracker(BaseCallbackHandler):
    """Collects input and output tokens across model calls. Reset it after reading."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_name = ""

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001 - LangChain callback signature
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) or {}
                self.input_tokens += int(usage.get("input_tokens", 0))
                self.output_tokens += int(usage.get("output_tokens", 0))
                self.model_name = (getattr(message, "response_metadata", {}) or {}).get("model_name", self.model_name)

    def take(self) -> tuple[int, int]:
        usage = (self.input_tokens, self.output_tokens)
        self.input_tokens = self.output_tokens = 0
        return usage


def _fingerprint(messages) -> str:
    return hashlib.sha256(json.dumps(messages, default=str, sort_keys=True).encode()).hexdigest()[:12]


class _Structured:
    def __init__(self, owner, schema: type[BaseModel]) -> None:
        self.owner, self.schema = owner, schema

    def invoke(self, messages, config=None):
        return self.owner._respond(self.schema, messages, config)


class ScriptedLLM:
    """Returns queued replies (or raises queued exceptions) and remembers the prompts it saw."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.prompts: list = []

    def with_structured_output(self, schema, method=None, **_kwargs):
        return _Structured(self, schema)

    def _respond(self, schema, messages, config):
        self.prompts.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class RecordingLLM:
    """Delegates to a real model and appends each exchange to a JSONL cassette."""

    def __init__(self, inner, cassette: Path) -> None:
        self.inner, self.cassette = inner, Path(cassette)
        self.cassette.parent.mkdir(parents=True, exist_ok=True)

    def with_structured_output(self, schema, method=None, **kwargs):
        self._structured = self.inner.with_structured_output(schema, method=method, **kwargs)
        return _Structured(self, schema)

    def _respond(self, schema, messages, config):
        reply = self._structured.invoke(messages, config=config) if config else self._structured.invoke(messages)
        entry = {"prompt": _fingerprint(messages), "schema": schema.__name__, "reply": reply.model_dump()}
        with open(self.cassette, "a", encoding="utf-8") as file:
            file.write(json.dumps(entry) + "\n")
        return reply


class ReplayLLM:
    """Plays a cassette back in order. `strict=True` also requires identical prompts."""

    def __init__(self, cassette: Path, strict: bool = False) -> None:
        with open(cassette, encoding="utf-8") as file:
            self.entries = [json.loads(line) for line in file if line.strip()]
        self.position, self.strict = 0, strict

    def with_structured_output(self, schema, method=None, **_kwargs):
        return _Structured(self, schema)

    def _respond(self, schema, messages, config):
        if self.position >= len(self.entries):
            raise RuntimeError("cassette exhausted: the replayed run asked for more model calls than were recorded")
        entry = self.entries[self.position]
        self.position += 1
        if self.strict and entry["prompt"] != _fingerprint(messages):
            raise RuntimeError(f"prompt {self.position} differs from the recording")
        return schema.model_validate(entry["reply"])
