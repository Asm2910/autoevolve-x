"""Ask the project a question. Answers come from its own documents, reports and memory, with sources.

  python -m scripts.ask "Why is xgboost the served model?"
  python -m scripts.ask "What did the evaluation find about missed fraud?" --llm     # needs ANTHROPIC_API_KEY
  python -m scripts.ask "..." --embedder hashing                                      # fuse word and embedding search

Without --llm the answer is the best-matching passages with their sources. With --llm a model writes a short
answer from those passages and the code checks its citations and numbers.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from agentkit.budget import Budget
from agentkit.tracing import Observer, TraceLog
from memory.embedding import get_embedder
from memory.store import MemoryStore
from models.training import PROCESSED_DIR, PROJECT_ROOT, load_config
from research.agent import ResearchAgent
from research.corpus import collect
from research.retriever import Retriever
from scripts.run_agent import llm_model_name, load_env_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--llm", action="store_true", help="let a language model write the answer (checked in code)")
    parser.add_argument("--embedder", choices=["none", "hashing", "minilm"], default="none")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    config = load_config()
    platform = config.get("agent_platform", {})
    budget = Budget.from_config(platform, PROJECT_ROOT)
    trace = PROJECT_ROOT / platform.get("trace_dir", "memory/traces") / f"research-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    llm = None
    if args.llm:
        load_env_file()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set. Put it in .env or export it, or run without --llm.")
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=llm_model_name(config), max_tokens=700, timeout=60, max_retries=2)

    store = MemoryStore()
    try:
        chunks = collect(PROJECT_ROOT, PROCESSED_DIR, store)
    finally:
        store.close()
    embedder = None if args.embedder == "none" else get_embedder(args.embedder)
    agent = ResearchAgent(Retriever(chunks, embedder), llm, budget, Observer(TraceLog(trace)), k=args.k)
    answer = agent.ask(args.question)

    print(f"\n{answer.answer}\n")
    if answer.ungrounded:
        print(f"Warning: numbers in the answer not found in the cited passages: {', '.join(answer.ungrounded)}")
    for s in answer.sources:
        print(f"  {s['label'] or '-'}  {s['source']}" + (f" > {s['section']}" if s["section"] else ""))
    print(f"\n({answer.mode}; {len(chunks)} passages searched)")


if __name__ == "__main__":
    main()
