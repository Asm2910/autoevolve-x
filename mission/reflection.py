"""Reflection: turn what happened into lessons the next mission can use, and decide whether to go again.

Every lesson states the evidence it rests on (a number, an interval or a verdict from deterministic
code). Nothing here is written by a language model, so a lesson cannot claim more than the run showed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Reflection:
    lessons: list[tuple[str, str]] = field(default_factory=list)  # (text, tags)
    next_steps: list[str] = field(default_factory=list)
    adopted: int = 0
    again: bool = False
    summary: str = ""


def reflect(results: dict, history: list[dict], round_number: int, max_rounds: int, last_round_adopted: int) -> Reflection:
    out = Reflection()
    data, evaluation, arena = results.get("data_review"), results.get("evaluate"), results.get("arena")

    if data:
        warnings = [f for f in data.get("findings", []) if f["severity"] in {"warning", "critical"}]
        if warnings:
            out.lessons.append((f"Data review ({data['verdict']}): " + "; ".join(f["title"] for f in warnings[:4]) + ".", "mission data"))
    if evaluation:
        weak = [f for f in evaluation.get("findings", []) if f["title"].startswith("Weaker on")]
        if weak:
            out.lessons.append(("Weak segments at the review cut-off: " + "; ".join(f["title"].removeprefix("Weaker on ") for f in weak[:3]) + ".", "mission evaluation"))
        out.next_steps += [h for h in evaluation.get("hypotheses", [])[:2]]
    if arena:
        out.lessons.append((f"Judge served {arena['winner']}: {arena['rationale']}", "mission judge"))

    adopted = [e for e in history if e["verdict"] == "adopt"]
    rejected = [e for e in history if e["verdict"] != "adopt"]
    out.adopted = len(adopted)
    for entry in adopted[:4]:
        out.lessons.append((f"Mission adopted: {entry['hypothesis']}. {entry['reason']}", "mission adopt"))
    for entry in rejected[:3]:
        out.lessons.append((f"Mission rejected: {entry['hypothesis']}. {entry['reason']}", "mission reject"))

    out.again = bool(last_round_adopted) and round_number < max_rounds
    if out.again:
        out.next_steps.insert(0, "The last round found an improvement, so another round is worth running.")
    elif not adopted:
        out.next_steps.append("No change beat the champion by a clear margin: more data or new features may help more than tuning.")
    out.summary = (f"{round_number} round(s); {len(adopted)} change(s) adopted, {len(rejected)} rejected."
                   if history else "No improvement experiments ran.")
    return out
