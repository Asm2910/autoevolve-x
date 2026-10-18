"""The mission report: what was asked, what the plan was, what each step found, and what to do next."""

from __future__ import annotations


def mission_payload(state: dict) -> dict:
    results = state.get("results", {})
    improve = results.get("improve", {})
    return {
        "mission": state["mission"], "plan": state.get("plan"), "halted": state.get("halted", ""),
        "timeline": state.get("timeline", []), "errors": state.get("errors", []),
        "rounds": state.get("round", 0), "history": state.get("history", []),
        "start_pr_auc": improve.get("start_pr_auc", results.get("evaluate", {}).get("pr_auc")), "end_pr_auc": improve.get("end_pr_auc"),
        "champion": improve.get("champion"), "served_model": results.get("arena", {}).get("winner"),
        "reflection": state.get("reflection", {}),
    }


def mission_markdown(state: dict) -> str:
    p = mission_payload(state)
    m, ref = p["mission"], p["reflection"]
    lines = ["# Mission report", "", f"**Goal:** {m['goal']}  ", f"**Depth:** {m['depth']}", ""]
    if p["halted"]:
        lines += [f"> **Halted:** {p['halted']}", ""]
    lines += ["## Plan", "", (p["plan"] or {}).get("rationale", ""), "", "| Step | Status | Seconds |", "|---|---|---|"]
    lines += [f"| {t['step']} | {t['status']} | {t['seconds']} |" for t in p["timeline"]]
    if p["errors"]:
        lines += ["", "Errors (the mission carried on):", *[f"- {e}" for e in p["errors"]]]
    results = state.get("results", {})
    if results.get("data_review"):
        lines += ["", f"## Data review: {results['data_review']['verdict']}", ""]
        lines += [f"- [{f['severity']}] {f['title']}" for f in results["data_review"]["findings"] if f["severity"] != "info"] or ["- No problems found."]
    if results.get("evaluate"):
        e = results["evaluate"]
        lines += ["", f"## Evaluation: {e['verdict']} (validation PR-AUC {e.get('pr_auc', float('nan')):.3f})", ""]
        lines += [f"- [{f['severity']}] {f['title']}" for f in e["findings"] if f["severity"] != "info"] or ["- No weaknesses found."]
    if results.get("arena"):
        lines += ["", f"## Judge: serve {results['arena']['winner']}", "", results["arena"]["rationale"]]
    if p["history"]:
        lines += ["", f"## Improvement ({p['rounds']} round(s))", ""]
        if p["start_pr_auc"] is not None and p["end_pr_auc"] is not None:
            lines += [f"Validation PR-AUC {p['start_pr_auc']:.3f} to {p['end_pr_auc']:.3f}.", ""]
        lines += [f"- [{h['verdict']}] {h['hypothesis']}: {h['reason']}" for h in p["history"]]
    if ref:
        lines += ["", "## Reflection", "", ref.get("summary", ""), ""]
        lines += [f"- {t}" for t in ref.get("lessons", [])]
        if ref.get("next_steps"):
            lines += ["", "Next steps:", *[f"- {t}" for t in ref["next_steps"]]]
    lines += ["", "_The sealed test split was not used. Confirm a champion on it once, at the end._", ""]
    return "\n".join(lines)
