"""Mission control: describe a goal, see the plan, run it, watch it, stop it."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from dashboard import control
from dashboard.theme import CSS, FONT_IMPORT

st.set_page_config(page_title="Mission control", page_icon=None, layout="wide")
st.html(f"<style>{FONT_IMPORT}{CSS}</style>")
st.title("Mission control")
st.caption("The planner chooses the steps from the depth. It does not read the goal text: the goal is recorded in the report.")

active = control.current()

with st.form("mission"):
    goal = st.text_input("Goal", "Improve fraud detection and explain what changed", disabled=active is not None)
    left, mid, right, last = st.columns(4)
    dataset = left.selectbox("Dataset", control.DATASETS, help="ieee is real e-commerce fraud data, not telecom")
    depth = mid.selectbox("Depth", list(control.DEPTHS), index=1)
    rounds = right.number_input("Rounds (0 = depth default)", 0, 5, 0)
    iterations = last.number_input("Experiments per round (0 = default)", 0, 12, 0)
    c1, c2 = st.columns([1, 1])
    preview = c1.form_submit_button("Preview plan")
    launch = c2.form_submit_button("Start mission", type="primary", disabled=active is not None)

try:
    plan = control.preview_plan(goal, depth, rounds or None, iterations or None)
except ValueError as err:
    plan = None
    st.error(str(err))

if plan and (preview or launch):
    st.write(plan["rationale"])
    st.table([{"step": s["name"], "why": s["why"], "settings": ", ".join(f"{k}={v}" for k, v in s["params"].items())} for s in plan["steps"]])

if plan and launch:
    try:
        run = control.start(goal, depth, dataset, rounds or None, iterations or None)
        st.success(f"Started mission {run.id} (process {run.pid}).")
        st.rerun()
    except RuntimeError as err:
        st.warning(str(err))


@st.fragment(run_every=3)
def live() -> None:
    run = control.current()
    if run is None:
        st.info("No mission is running.")
        return
    st.write(f"**Running:** {run.goal} ({run.depth}, {run.dataset}), started {run.started}")
    if st.button("Stop mission", key="stop"):
        control.stop()
        st.warning("Stop requested. The mission ends at its next step and still writes its reflection.")
    st.code(control.tail(run.log) or "(waiting for output)", language="text")


live()

st.subheader("Latest report")
view = st.radio("Dataset", control.DATASETS, horizontal=True, label_visibility="collapsed", key="report_dataset")
report = control.latest_report(view)
if report is None:
    st.caption("No mission report for this dataset yet.")
else:
    st.markdown(report[0])

with st.expander("Earlier runs"):
    st.table([{"id": r.id, "goal": r.goal, "depth": r.depth, "dataset": r.dataset, "started": r.started}
              for r in control.history()])
