"""The rule that decides whether a candidate replaces the champion.

This is plain code on purpose. A language model may propose experiments, but only this
rule, applied to measured results, decides whether anything is adopted.
"""

from __future__ import annotations


def decide(result: dict, min_gain: float) -> tuple[str, str]:
    """Return (verdict, reason). Adopt only a clear, meaningful improvement."""
    low, high = result["diff_ci"]
    mean = result["diff_mean"]
    interval_text = f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]"
    if low > 0 and mean >= min_gain:
        return "adopt", f"validation PR-AUC change {interval_text}: significant and at least {min_gain:+.3f}"
    if low > 0:
        return "reject", f"validation PR-AUC change {interval_text}: significant but smaller than {min_gain:+.3f}"
    if high < 0:
        return "reject", f"validation PR-AUC change {interval_text}: significantly worse"
    return "reject", f"validation PR-AUC change {interval_text}: interval includes zero, no clear difference"