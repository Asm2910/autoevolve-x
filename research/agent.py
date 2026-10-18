"""The research agent: answer a question from the project's own records, with sources, or say it cannot.

  1. Retrieve the best-matching chunks (BM25, optionally fused with embeddings).
  2. Without a language model: return the best passages as an extractive answer, each with its source.
  3. With a language model: ask it to answer ONLY from the numbered passages and to cite them, then check
     the answer in code:
       - every cited id must be one of the passages it was shown,
       - every number in the answer must appear in a cited passage (otherwise it is listed as ungrounded),
       - no citation at all means the answer is withheld.
  4. If nothing matches, say so. It does not guess.

Retrieved text is untrusted: it is sanitised and delimited like any outside text before it reaches a prompt,
so a note that says "ignore your instructions" is replaced by a placeholder. The model proposes an answer;
deterministic code decides whether it is allowed through.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from agentkit.budget import Budget
from agentkit.context import ContextBuilder, versioned_prompt
from agentkit.guardrails import UNTRUSTED_NOTICE, clean_model_text, sanitize_untrusted
from agentkit.llm import UsageTracker
from agentkit.tracing import Observer
from observability import agent_metrics as metrics
from research.corpus import Chunk
from research.retriever import Retriever

NOT_FOUND = "I could not find this in the project's records."
NUMBER = re.compile(r"(?<![\w.])-?\+?\d[\d,]*\.?\d*%?")

SYSTEM = (
    "You answer questions about one machine-learning project using only the numbered passages you are given.\n"
    "Rules:\n- Use only facts stated in the passages. Do not add outside knowledge.\n"
    "- Cite the passages you used by id in `cited` (for example S1, S3).\n"
    "- Quote numbers exactly as they appear.\n"
    "- If the passages do not contain the answer, set insufficient=true and leave `answer` empty.\n"
    "- Keep the answer to a few sentences."
)


class LLMAnswer(BaseModel):
    answer: str = Field(default="", description="Short answer grounded in the passages")
    cited: list[str] = Field(default_factory=list, description="Ids of the passages used, such as S1")
    insufficient: bool = Field(default=False, description="True if the passages do not contain the answer")


class Answer(BaseModel):
    question: str
    answer: str
    mode: str  # extractive | llm | not_found | withheld
    sources: list[dict] = Field(default_factory=list)
    ungrounded: list[str] = Field(default_factory=list)
    flagged_inputs: int = 0


def numbers_in(text: str) -> set[str]:
    return {n.replace(",", "").lstrip("+").rstrip(".") for n in NUMBER.findall(text)}


class ResearchAgent:
    def __init__(self, retriever: Retriever, llm=None, budget: Budget | None = None, observer: Observer | None = None,
                 k: int = 5, max_context_tokens: int = 3000) -> None:
        self.retriever = retriever
        self.structured = llm.with_structured_output(LLMAnswer, method="json_schema") if llm is not None else None
        self.budget = budget or Budget.unlimited()
        self.observer = observer or Observer()
        self.k, self.max_context_tokens = k, max_context_tokens
        self.usage = UsageTracker()
        self.prompt = versioned_prompt("research.system", SYSTEM + "\n\n" + UNTRUSTED_NOTICE)

    # -- public ----------------------------------------------------------------------------------
    def ask(self, question: str) -> Answer:
        self.budget.check()
        clean = sanitize_untrusted(question, max_chars=300)
        with self.observer.step("research", kind="llm" if self.structured else "deterministic") as outcome:
            hits = self.retriever.search(clean.text, self.k)
            outcome["retrieved"] = len(hits)
            if clean.suspicious or not hits:
                answer = Answer(question=clean.text, answer=NOT_FOUND, mode="not_found", flagged_inputs=len(clean.flags))
            elif self.structured is None:
                answer = self._extractive(clean.text, hits)
            else:
                answer = self._generate(clean.text, hits)
            outcome["mode"] = answer.mode
            return answer

    # -- modes -----------------------------------------------------------------------------------
    @staticmethod
    def _source(chunk: Chunk, score: float, label: str = "") -> dict:
        return {"label": label, "id": chunk.id, "source": chunk.source, "section": chunk.section,
                "kind": chunk.kind, "score": round(score, 3), "text": chunk.text[:400]}

    def _extractive(self, question: str, hits: list[tuple[Chunk, float]]) -> Answer:
        top = hits[:3]
        lines = [f"- {c.text.strip().replace(chr(10), ' ')[:300]} [{c.source}]" for c, _ in top]
        return Answer(question=question, answer="Closest passages in the project's records:\n" + "\n".join(lines),
                      mode="extractive", sources=[self._source(c, s, f"S{i}") for i, (c, s) in enumerate(top, 1)])

    def _generate(self, question: str, hits: list[tuple[Chunk, float]]) -> Answer:
        labels = {f"S{i}": chunk for i, (chunk, _) in enumerate(hits, 1)}
        flagged = 0
        builder = ContextBuilder(self.max_context_tokens)
        for i, (label, chunk) in enumerate(labels.items(), 1):
            clean = sanitize_untrusted(chunk.text, max_chars=700)
            flagged += bool(clean.flags)
            builder.add(label, f"[{label}] ({chunk.source}) {clean.text}", priority=i)
        builder.add("question", f"Question: {question}", required=True)
        reply: LLMAnswer = self.structured.invoke(
            [("system", self.prompt.text), ("human", f'<untrusted_data label="passages">\n{builder.build().text}\n</untrusted_data>')],
            config={"callbacks": [self.usage]},
        )
        tokens_in, tokens_out = self.usage.take()
        self.budget.charge_llm(tokens_in, tokens_out)
        metrics.LLM_REQUESTS.labels("research", self.usage.model_name or "unknown").inc()
        metrics.LLM_TOKENS.labels("research", "input").inc(tokens_in)
        metrics.LLM_TOKENS.labels("research", "output").inc(tokens_out)

        cited = [c for c in dict.fromkeys(reply.cited) if c in labels]
        text = clean_model_text(reply.answer, 800)
        if reply.insufficient or not text:
            return Answer(question=question, answer=NOT_FOUND, mode="not_found", flagged_inputs=flagged)
        if not cited:
            return Answer(question=question, answer="The model answered without citing any passage, so the answer was withheld.",
                          mode="withheld", flagged_inputs=flagged,
                          sources=[self._source(c, s, f"S{i}") for i, (c, s) in enumerate(hits[:3], 1)])
        evidence = " ".join(labels[c].text for c in cited)
        known = numbers_in(evidence)
        ungrounded = sorted(n for n in numbers_in(text) if n not in known)
        scores = {chunk.id: score for chunk, score in hits}
        return Answer(question=question, answer=text, mode="llm", ungrounded=ungrounded, flagged_inputs=flagged,
                      sources=[self._source(labels[c], scores[labels[c].id], c) for c in cited])
