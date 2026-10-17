"""Charts as plain HTML and CSS (no chart library, no JavaScript): every mark is a positioned element.

Each function returns an HTML string for st.html. Values are positioned by percentage of the axis,
so the layout stays correct at any width, and rows can wrap to any height without breaking alignment.
"""

from __future__ import annotations

import math
from html import escape

from dashboard.data import group_of, humanize, pretty_date
from dashboard.theme import FALLBACK_GROUP, GROUP_COLORS

MINUS = "\u2212"


def signed(value: float, digits: int = 3) -> str:
    text = f"{abs(value):.{digits}f}"
    return f"+{text}" if value > 0 else (f"{MINUS}{text}" if value < 0 else text)


def pos(value: float, low: float, high: float) -> float:
    return max(0.0, min(100.0, (value - low) / (high - low) * 100))


def color_of(group: str) -> str:
    return GROUP_COLORS.get(group, FALLBACK_GROUP)


def section(title: str, blurb: str, body: str) -> str:
    return (f'<section class="sec"><div class="rail"><h2>{escape(title)}</h2><p>{escape(blurb)}</p></div>'
            f'<div class="body">{body}</div></section>')


def empty(message: str, command: str = "") -> str:
    tail = f" Run <code>{escape(command)}</code>." if command else ""
    return f'<p class="empty">{escape(message)}{tail}</p>'


# ---- masthead -------------------------------------------------------------------------------
def masthead(ledger: list[dict]) -> str:
    adopted = sum(1 for e in ledger if e["verdict"] == "adopt")
    if ledger:
        headline = f"The agent has tried {len(ledger)} ideas and kept {adopted}."
    else:
        headline = "The agent has not run any experiments yet."
    return (
        '<header class="mast"><div class="name">AutoEvolve-X</div>'
        f"<h1>{escape(headline)}</h1>"
        "<p>A language model proposes changes to a fraud-risk model. Plain code runs each one and applies a fixed "
        "statistical rule to decide whether to believe it. Everything below is read from the agent's own records.</p>"
        "</header>"
    )


# ---- final test ---------------------------------------------------------------------------------
def _ticks(high: float) -> list[float]:
    step = 0.1 if high > 0.25 else 0.05
    return [round(i * step, 3) for i in range(int(high / step) + 1)]


def final_test_section(final: dict | None) -> str:
    rail = ("Final test", "Scored once, on snapshots the agent never saw.")
    if not final:
        return section(*rail, empty("The champion has not been scored on the test snapshots yet.",
                                    "python -m scripts.confirm_champion"))
    m = final["metrics"]
    champ, base = m["champion"], m["baseline"]
    rows, positives = m["n_test_rows"], m["n_test_positives"]
    chance = positives / rows
    low_d, high_d = m["diff_pr_auc"]["ci95"]
    high = max(0.4, math.ceil(champ["test_ci95"]["pr_auc"][1] * 1.3 / 0.1) * 0.1)
    ticks = _ticks(high)

    def row(label: str, sub: str, entry: dict, cls: str) -> str:
        lo, hi = entry["test_ci95"]["pr_auc"]
        value = entry["test"]["pr_auc"]
        grid = "".join(f'<i class="g" style="left:{pos(t, 0, high):.2f}%"></i>' for t in ticks)
        return (
            f'<div class="rrow {cls}"><div class="rname">{escape(label)}<small>{escape(sub)}</small></div>'
            f'<div class="track">{grid}<b class="ci" style="left:{pos(lo, 0, high):.2f}%;'
            f'width:{pos(hi, 0, high) - pos(lo, 0, high):.2f}%"></b>'
            f'<u class="dot" style="left:{pos(value, 0, high):.2f}%"></u>'
            f'<span class="val" style="left:{pos(hi, 0, high):.2f}%">{value:.3f}</span></div></div>'
        )

    model = champ["spec"]["model"].replace("_", " ")
    verdict_text = (
        f"The champion beats a plain logistic regression by {signed(m['diff_pr_auc']['mean'])} "
        f"(95% interval {signed(low_d)} to {signed(high_d)}). The interval stays above zero, so the gap is real."
        if m["verdict"] == "confirmed"
        else f"The champion's lead over a plain logistic regression is {signed(m['diff_pr_auc']['mean'])} "
             f"(95% interval {signed(low_d)} to {signed(high_d)}). The interval includes zero, so the gap is not proven."
    )
    axis = "".join(f'<span style="left:{pos(t, 0, high):.2f}%">{t:.2f}</span>' for t in ticks)
    body = (
        f'<div class="headline"><span class="big">{champ["test"]["pr_auc"]:.3f}</span>'
        f'<p class="lede">Test PR-AUC of the champion {escape(model)} model on {rows:,} subscriber-months, '
        f"{positives:,} of them confirmed fraud. <span>{escape(verdict_text)}</span></p></div>"
        '<div class="ruler">'
        f'<div class="cw"><div class="chance" style="left:{pos(chance, 0, high):.2f}%">'
        f"<span>A random scorer gets {chance:.3f}</span></div></div>"
        + row("Champion", f"{model}, 95% interval", champ, "champ")
        + row("Baseline", "logistic regression, 95% interval", base, "base")
        + f'<div class="axis"><div></div><div class="scale">{axis}</div></div></div>'
        f'<p class="note">PR-AUC rewards putting real fraud near the top of the list, which is what an analyst '
        f"with limited review time needs. With fraud this rare, even a perfect-looking accuracy figure would be meaningless.</p>"
    )
    return section(*rail, body)


# ---- ledger: the forest plot ------------------------------------------------------------------------
def ledger_section(ledger: list[dict], min_gain: float = 0.01) -> str:
    rail = ("Experiment ledger",
            "Each line is one idea. The dot is its measured gain on validation data, the bar is the 95% interval.")
    if not ledger:
        return section(*rail, empty("No experiments are recorded for this dataset yet.",
                                    "python -m scripts.run_agent --iterations 4"))
    reach = max(max(abs(e["low"]), abs(e["high"]), abs(e["mean"])) for e in ledger)
    half = max(0.04, round(math.ceil(reach / 0.02) * 0.02, 2))
    low_ax, high_ax = -half, half
    step = 0.02
    ticks = [round(low_ax + i * step, 3) for i in range(round(2 * half / step) + 1)]

    scale = "".join(
        f'<span style="left:{pos(t, low_ax, high_ax):.2f}%">{signed(t, 2) if t else "0"}</span>' for t in ticks
    )
    scale += (f'<span class="bar" style="left:{pos(min_gain, low_ax, high_ax):.2f}%">'
              f"Adopt zone: gain of {min_gain:.2f} or more</span>")
    head = ('<ol class="ledger"><li class="lhead"><span></span><span></span>'
            f'<div class="scale">{scale}</div><span></span></li>')
    out = [head]
    for e in ledger:
        cls = "adopt" if e["verdict"] == "adopt" else "reject"
        verdict = "Adopted" if e["verdict"] == "adopt" else "Rejected"
        why = escape(e["conclusion"].split(":", 1)[-1].strip() or e["conclusion"])
        zone = pos(min_gain, low_ax, high_ax)
        out.append(
            f'<li class="lrow {cls}" title="{escape(e["hypothesis"])}">'
            f'<span class="idx">{e["id"]}</span>'
            f'<div><p class="hyp">{escape(e["hypothesis"])}</p>'
            f'<span class="why"><span class="verdict">{verdict}.</span> {why}</span></div>'
            f'<div class="plot"><i class="zone" style="left:{zone:.2f}%"></i>'
            f'<i class="zero" style="left:{pos(0, low_ax, high_ax):.2f}%"></i>'
            f'<b class="ci" style="left:{pos(e["low"], low_ax, high_ax):.2f}%;'
            f'width:{pos(e["high"], low_ax, high_ax) - pos(e["low"], low_ax, high_ax):.2f}%"></b>'
            f'<u class="dot" style="left:{pos(e["mean"], low_ax, high_ax):.2f}%"></u></div>'
            f'<div class="nums"><strong>{signed(e["mean"])}</strong>'
            f'<span>{signed(e["low"])} to {signed(e["high"])}</span></div></li>'
        )
    out.append("</ol>")
    adopted = sum(1 for e in ledger if e["verdict"] == "adopt")
    widest = max(e["high"] - e["low"] for e in ledger)
    tally = (
        f'<p class="tally"><b>{adopted} of {len(ledger)}</b> ideas were adopted. To count, an idea must have an '
        f"interval entirely above zero <i>and</i> a gain of at least {min_gain:.2f}. The widest interval here spans "
        f"{widest:.3f}, which is how much noise a single experiment carries.</p>"
    )
    return section(*rail, "".join(out) + tally)


# ---- where the signal comes from ---------------------------------------------------------------------
def signal_section(groups: list[tuple[str, float]] | None, top: list[dict] | None) -> str:
    rail = ("What drives the score", "Share of the model's decisions explained by each kind of feature.")
    if not groups:
        return section(*rail, empty("No explanation has been generated yet.", "python -m scripts.explain_model"))
    segments = "".join(
        f'<i style="flex:{share:.5f};background:{color_of(g)}" title="{escape(g)} {share:.1%}"></i>' for g, share in groups
    )
    keys = "".join(
        f'<span class="key"><span class="sw" style="background:{color_of(g)}"></span><b>{escape(g.capitalize())}</b>'
        f"<span>{share:.0%}</span></span>"
        for g, share in groups
    )
    bars = ""
    if top:
        peak = max(t["share"] for t in top) or 1.0
        rows = "".join(
            f'<div class="brow"><div class="nm"><span class="sw" style="background:{color_of(group_of(t["feature"]))}">'
            f'</span><em title="{escape(t["feature"])}">{escape(humanize(t["feature"]))}</em></div>'
            f'<div class="tr"><b style="width:{t["share"] / peak * 100:.1f}%;background:{color_of(group_of(t["feature"]))}">'
            f'</b></div><div class="v">{t["share"]:.1%}</div></div>'
            for t in top
        )
        bars = f'<div class="bars">{rows}</div>'
    return section(*rail, f'<div class="share">{segments}</div><div class="keys">{keys}</div>{bars}')


# ---- drift ---------------------------------------------------------------------------------------------
def drift_section(checks: list[dict], table: list[dict] | None, moderate: float = 0.10, major: float = 0.25) -> str:
    rail = ("Data drift", "Whether new snapshots still look like the data the model learned from.")
    if not checks and not table:
        return section(*rail, empty("No drift check has been recorded yet.", "python -m scripts.monitor_drift"))
    items = ""
    for c in checks:
        m = c["metrics"]
        ok = c["verdict"] == "stable"
        pill = f'<span class="pill {"ok" if ok else "warn"}">{"Stable" if ok else "Drift"}</span>'
        detail = (f"{m.get('n_major', 0)} of {m.get('n_features', 0)} features shifted a lot, "
                  f"{m.get('n_moderate', 0)} a little.")
        items += f'<li><span class="when">{escape(pretty_date(c["name"]))}</span><span>{pill}{escape(detail)}</span></li>'
    body = f'<ul class="checks">{items}</ul>' if items else ""
    if table:
        peak = max(max(r["psi"] for r in table) * 1.15, major * 1.15)
        peak = math.ceil(peak / 0.05) * 0.05
        guides = "".join(f'<i class="g" style="left:{pos(v, 0, peak):.2f}%"></i>' for v in (moderate, major))
        labels = (f'<span style="left:{pos(moderate, 0, peak):.2f}%;transform:translateX(-50%)">{moderate:.2f} noticeable</span>'
                  f'<span style="left:{pos(major, 0, peak):.2f}%;transform:translateX(-50%)">{major:.2f} major</span>')
        rows = "".join(
            f'<div class="brow"><div class="nm"><span class="sw" style="background:{color_of(group_of(r["feature"]))}">'
            f'</span><em title="{escape(r["feature"])}">{escape(humanize(r["feature"]))}</em></div>'
            f'<div class="tr">{guides}<b style="width:{pos(r["psi"], 0, peak):.2f}%;'
            f'background:{color_of(group_of(r["feature"]))}"></b></div><div class="v">{r["psi"]:.3f}</div></div>'
            for r in table
        )
        body += (f'<div class="gh"><div>Most shifted features, newest snapshot</div>'
                 f'<div class="scale">{labels}</div><div></div></div><div class="bars">{rows}</div>')
        if max(r["psi"] for r in table) < moderate:
            body += '<p class="note">Even the most shifted feature sits left of the noticeable line.</p>'
    return section(*rail, body)


# ---- memory ---------------------------------------------------------------------------------------------
def memory_rail(title: str, blurb: str) -> str:
    return f'<div class="rail"><h2>{escape(title)}</h2><p>{escape(blurb)}</p></div>'


def lesson_list(items: list[dict]) -> str:
    if not items:
        return empty("Nothing matches. Try a feature group such as usage, or a word like drift.")
    rows = ""
    for item in items:
        text, meta = item["text"], pretty_date(item.get("created_at", "")[:10])
        if item.get("tags"):
            meta += f", {item['tags']}"
        rows += f"<li><p>{escape(text)}</p><small>{escape(meta)}</small></li>"
    return f'<ul class="mem">{rows}</ul>'


def footer(final: dict | None) -> str:
    if not final:
        return '<footer class="foot">Nothing has been confirmed on the test snapshots yet.</footer>'
    sha = (final.get("dataset_sha256") or "")[:12]
    commit = final.get("git_commit") or "unknown"
    dirty = " with uncommitted changes" if final.get("git_dirty") else ""
    return (f'<footer class="foot">The final test ran on {escape(pretty_date(final["created_at"][:10]))} against '
            f"dataset {escape(sha)}, code version {escape(commit)}{dirty}. It can run only once per dataset.</footer>")