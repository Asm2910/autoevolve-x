"""Context engineering: decide what the model sees, and keep it small and stable.

  ContextBuilder   fits prioritised sections into a token budget. Stable instructions come
                   first (so the provider's prompt cache can reuse them); evidence follows;
                   the least important evidence is trimmed or dropped when space runs out.
  compact_history  keeps the newest experiments in full and folds older ones into one
                   deterministic summary line, so the prompt does not grow without limit.
  ContextPolicy    each agent receives only the context keys it needs.
  versioned_prompt every prompt has an id derived from its text, recorded in the trace, so a
                   result can always be traced to the exact wording that produced it.

Token counts are an estimate (about four characters per token). They exist to stay inside a
budget, not to bill anyone; the real counts come back from the API and go to the Budget.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


class ContextTooLarge(ValueError):
    """The required sections alone do not fit in the budget."""


@dataclass
class Section:
    name: str
    text: str
    priority: int = 5  # 1 = most important; larger numbers are trimmed first
    required: bool = False


@dataclass
class BuiltContext:
    text: str
    tokens: int
    included: list[str] = field(default_factory=list)
    trimmed: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


class ContextBuilder:
    def __init__(self, budget_tokens: int) -> None:
        self.budget_tokens = budget_tokens
        self.sections: list[Section] = []

    def add(self, name: str, text: str, priority: int = 5, required: bool = False) -> ContextBuilder:
        self.sections.append(Section(name, text, priority, required))
        return self

    def build(self) -> BuiltContext:
        used = sum(estimate_tokens(s.text) for s in self.sections if s.required)
        if used > self.budget_tokens:
            raise ContextTooLarge(f"required sections need ~{used} tokens, budget is {self.budget_tokens}")
        kept: dict[str, str] = {s.name: s.text for s in self.sections if s.required}
        trimmed, dropped = [], []
        for section in sorted((s for s in self.sections if not s.required), key=lambda s: s.priority):
            need = estimate_tokens(section.text)
            room = self.budget_tokens - used
            if need <= room:
                kept[section.name] = section.text
                used += need
            elif room >= 40:  # keep the start of it
                kept[section.name] = section.text[: room * 4].rsplit("\n", 1)[0] + "\n[trimmed to fit]"
                used += estimate_tokens(kept[section.name])
                trimmed.append(section.name)
            else:
                dropped.append(section.name)
        order = [s.name for s in self.sections if s.name in kept]
        text = "\n\n".join(kept[name] for name in order)
        return BuiltContext(text, estimate_tokens(text), order, trimmed, dropped)


def compact_history(items: list[dict], keep_recent: int = 8, verdict_key: str = "verdict") -> tuple[list[dict], str]:
    """Newest items in full, older ones summarised. Returns (recent_items, summary_line)."""
    if len(items) <= keep_recent:
        return items, ""
    older, recent = items[:-keep_recent], items[-keep_recent:]
    counts = Counter(item.get(verdict_key, "unknown") for item in older)
    parts = ", ".join(f"{n} {verdict}" for verdict, n in sorted(counts.items()))
    return recent, f"Earlier experiments ({len(older)} older): {parts}."


@dataclass(frozen=True)
class ContextPolicy:
    """The context keys one agent is allowed to see."""

    agent: str
    allowed_keys: frozenset[str]

    def apply(self, context: dict) -> dict:
        return {key: value for key, value in context.items() if key in self.allowed_keys}


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str

    @property
    def version(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:8]

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"


def versioned_prompt(name: str, text: str) -> Prompt:
    return Prompt(name, text)
