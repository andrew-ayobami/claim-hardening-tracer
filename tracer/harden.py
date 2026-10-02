"""Hardening step: mark each match as hedged or stated as fact, and find where the doubt was lost.

A match is hedged if a word or phrase from hedges.txt appears in its context:
the matched sentence alone, the sentence plus one either side, or the whole
line. The hardening point is the first match stated as fact that comes after
at least one hedged match. The actions after it are the computer-use sessions
whose goal matches the claim.

Only strong matches count here: weak matches are often same-topic chatter.
"""

import re
from pathlib import Path

from tracer.match import parse_time

HEDGES_FILE = Path(__file__).with_name("hedges.txt")
SCOPE = "neighbours"  # "sentence", "neighbours" or "line"
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def load_hedges(path=HEDGES_FILE):
    """Return a pattern matching any hedge phrase, and a map back to each phrase as written in the file."""
    phrases = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    phrases = sorted({p for p in phrases if p and not p.startswith("#")}, key=len, reverse=True)
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b", re.IGNORECASE)
    return pattern, {p.lower(): p for p in phrases}


def context(text, span, scope=SCOPE):
    """Return the text around a match that's checked for hedges. It never crosses a line break."""
    start, end = span
    if scope == "sentence":
        return text[start:end]
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end == -1 else line_end
    line = text[line_start:line_end]
    if scope == "line":
        return line

    # One sentence either side of the matched span, within the line.
    bounds = [0] + [m.end() for m in SENTENCE_BREAK.finditer(line)] + [len(line)]
    first = max(i for i in range(len(bounds) - 1) if bounds[i] <= start - line_start)
    last = max(i for i in range(len(bounds) - 1) if bounds[i] < max(end - line_start, 1))
    return line[bounds[max(first - 1, 0)]:bounds[min(last + 2, len(bounds) - 1)]]


def harden(matches, hedges=None, scope=SCOPE):
    """Mark each match hedged or fact, and return the hardening analysis.

    Adds "stance" ("hedged" or "fact") and "hedges" (the phrases found) to every
    match, and returns a dict with the hardening point, the hedged matches
    before it and the actions after it.
    """
    pattern, canonical = hedges or load_hedges()
    for m in matches:
        found = pattern.findall(context(m["record"]["text"], m["span"], scope))
        m["hedges"] = sorted({canonical.get(f.lower(), f) for f in found}, key=str.lower)
        m["stance"] = "hedged" if found else "fact"

    strong = [m for m in matches if m["strong"]]
    point, hedged_before = None, []
    for m in strong:
        if m["stance"] == "hedged":
            hedged_before.append(m)
        elif hedged_before:
            point = m
            break

    actions = []
    if point:
        after = parse_time(point["record"]["time"])
        actions = [m for m in strong if m["record"]["channel"] == "computer use"
                   and parse_time(m["record"]["time"]) > after]
    return {
        "origin": strong[0] if strong else None,
        "hardening_point": point,
        "hedged_before": hedged_before,
        "actions": actions,
    }
