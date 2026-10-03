"""Hardening step: mark each match's stance, and find where the doubt was lost.

Each match gets one stance, checked in this order:
- question: the matched sentence asks something ("isn't there an hour left?")
- disputed: the matched sentence denies or corrects the claim, using a phrase
  from disputes.txt ("does not exist", "mistakenly"), unless that phrase is
  part of the claim itself
- hedged: a phrase from hedges.txt appears in the match's context, which is the
  whole line by default (see SCOPE)
- fact: none of the above

The hardening point is the first close restatement of the claim (a core match)
stated as fact within WINDOW of a hedged match, in chat or memory. Computer-use
sessions can't be the hardening point: a session goal is a plan that takes its
premise as given ("Check if Pages has deployed the fix"), not a statement. Hedges count from any strong
match, since doubt is often voiced in looser wording than the claim ("Reddit
likely isn't reachable"). A fact hours after the last hedge is a new episode
rather than doubt being dropped, hence the window. Questions and disputes are
neither doubt nor fact.
The actions after the hardening point are the computer-use sessions whose goal
matches the claim. Weak matches play no part: they are often same-topic chatter.
"""

import re
from datetime import timedelta
from pathlib import Path

from tracer.match import origin, parse_time

WINDOW = timedelta(minutes=30)  # a hardening fact must follow a hedge within this

HEDGES_FILE = Path(__file__).with_name("hedges.txt")
DISPUTES_FILE = Path(__file__).with_name("disputes.txt")
# How much text around a match is checked for hedges. The hedge is often in the
# sentence after the match ("I clicked Submit... I think it went through"), and
# on the demo and the real test cases "line" did best.
SCOPE = "line"  # "sentence", "neighbours" or "line"
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
QUESTION = re.compile(r"\?[\s\"'”’*_)\]]*$")


def load_phrases(path):
    """Return a pattern matching any phrase in the file, and a map back to each phrase as written."""
    phrases = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    phrases = sorted({p for p in phrases if p and not p.startswith("#")}, key=len, reverse=True)
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b", re.IGNORECASE)
    return pattern, {p.lower(): p for p in phrases}


def load_hedges():
    return load_phrases(HEDGES_FILE)


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


def found_phrases(pattern, canonical, text, ignore=()):
    found = {canonical.get(f.lower(), f) for f in pattern.findall(text)}
    return sorted((f for f in found if f.lower() not in ignore), key=str.lower)


def harden(matches, claim="", hedges=None, scope=SCOPE):
    """Mark each match's stance and return the hardening analysis.

    Adds "stance" (question, disputed, hedged or fact), "hedges" and "disputes"
    (the phrases found) to every match, and returns a dict with the origin, the
    hardening point, the hedged matches before it and the actions after it.
    """
    hedge_pattern, hedge_names = hedges or load_hedges()
    dispute_pattern, dispute_names = load_phrases(DISPUTES_FILE)
    in_claim = {p for p in dispute_names if re.search(rf"\b{re.escape(p)}\b", claim, re.IGNORECASE)}
    for m in matches:
        sentence = m["excerpt"]
        m["hedges"] = found_phrases(hedge_pattern, hedge_names, context(m["record"]["text"], m["span"], scope))
        m["disputes"] = found_phrases(dispute_pattern, dispute_names, sentence, in_claim)
        if QUESTION.search(sentence):
            m["stance"] = "question"
        elif m["disputes"]:
            m["stance"] = "disputed"
        elif m["hedges"]:
            m["stance"] = "hedged"
        else:
            m["stance"] = "fact"

    strong = [m for m in matches if m["strong"]]
    point, hedged = None, []
    for m in strong:
        t = parse_time(m["record"]["time"])
        if m["stance"] == "hedged":
            hedged.append(m)
        elif (m["stance"] == "fact" and m["core"] and m["record"]["channel"] != "computer use" and hedged
              and t - parse_time(hedged[-1]["record"]["time"]) <= WINDOW):
            point = m
            break
    # The doubt that was dropped: the hedged matches in the window before the point.
    hedged_before = [h for h in hedged if point and
                     parse_time(point["record"]["time"]) - parse_time(h["record"]["time"]) <= WINDOW]

    actions = []
    if point:
        after = parse_time(point["record"]["time"])
        actions = [m for m in strong if m["record"]["channel"] == "computer use"
                   and parse_time(m["record"]["time"]) > after]
    return {
        "origin": origin(matches),
        "hardening_point": point,
        "hedged_before": hedged_before,
        "actions": actions,
    }
