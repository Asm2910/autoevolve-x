"""Ask the project: answers come from its own documents, reports and memory, with sources."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from agentkit.tracing import Observer
from dashboard import data
from dashboard.theme import CSS, FONT_IMPORT
from models.training import PROCESSED_DIR, PROJECT_ROOT
from research.agent import ResearchAgent
from research.corpus import collect
from research.retriever import Retriever

st.set_page_config(page_title="Ask the project", page_icon=None, layout="wide")
st.html(f"<style>{FONT_IMPORT}{CSS}</style>")
st.title("Ask the project")
st.caption("Searches the README, docs, reports and the agent's memory. Without a model it shows the closest passages; "
           "run `python -m scripts.ask --llm` for a written, checked answer.")


@st.cache_resource(ttl=300)
def retriever() -> Retriever:
    return Retriever(collect(PROJECT_ROOT, PROCESSED_DIR, data.open_store()))


question = st.text_input("Question", placeholder="Why is xgboost the served model?")
if question.strip():
    answer = ResearchAgent(retriever(), observer=Observer()).ask(question)
    st.write(answer.answer)
    for s in answer.sources:
        with st.expander(f"{s['label']}  {s['source']}" + (f" > {s['section']}" if s["section"] else "")):
            st.write(s["text"])
