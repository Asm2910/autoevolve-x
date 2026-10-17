"""AutoEvolve-X dashboard.   Run from the project root:   streamlit run dashboard/app.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from dashboard import charts, data
from dashboard.theme import CSS, FONT_IMPORT
from models.training import load_config

st.set_page_config(page_title="AutoEvolve-X", page_icon=None, layout="wide")
st.html(f"<style>{FONT_IMPORT}{CSS}</style>")

config = load_config()
store = data.open_store()

final = data.latest_final_test(store)
ledger = data.agent_ledger(store, final["dataset_sha256"] if final else None)

importance = data.read_csv("feature_importance.csv")
groups = data.importance_by_group(importance) if importance is not None else None
top = importance.sort_values("share", ascending=False).head(8).to_dict("records") if importance is not None else None

drift_table = data.read_csv("drift_latest.csv")
drift_rows = drift_table.head(10).to_dict("records") if drift_table is not None else None

st.html(charts.masthead(ledger))
st.html(charts.final_test_section(final))
st.html(charts.ledger_section(ledger, float(config["agent"]["min_gain"])))
st.html(charts.signal_section(groups, top))
st.html(charts.drift_section(data.drift_checks(store), drift_rows))

with st.container(key="memory"):
    rail, content = st.columns([1, 4])
    rail.html(charts.memory_rail("What the agent remembers", "Verified lessons from past runs. Search by word or by meaning."))
    with content:
        query = st.text_input("Search the agent's memory", placeholder="Search lessons, for example: usage features",
                              label_visibility="collapsed")
        if query.strip():
            hits = store.search(query.strip(), limit=6)
            found = [h["item"] if h["type"] == "lessons" else
                     {"text": h["item"]["conclusion"] or h["item"]["hypothesis"], "created_at": h["item"]["created_at"],
                      "tags": h["item"]["kind"]} for h in hits]
        else:
            found = data.lessons(store, limit=6)
        st.html(charts.lesson_list(found))

st.html(charts.footer(final))