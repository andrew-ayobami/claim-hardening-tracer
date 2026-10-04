"""Check the tracer against hand-labelled answer keys.

An answer key names a records file and, for each claim, the record ids the
tracer should find (grouped by role) and any it must not. A case with a "day"
is searched within that village day.

demo_data/expected.json is complete: every record is labelled, so every check
is strict (all appearances found, the near miss left out, the right origin,
stance labels, hardening point and actions).

test_cases.json is partial: it lists key records found by reading the chat,
not every appearance (most memory restatements are missing from it). So a case
passes if the hardening point is right, meaning a match stated as fact within
TOLERANCE of the labelled one (or no hardening point when none is expected),
and no record it must not match is matched. Missed records, the origin and
stance labels are reported for review but don't fail the case.
"""

import json
from datetime import timedelta
from pathlib import Path

from tracer.harden import SCOPE, harden, load_hedges
from tracer.judge import apply_labels
from tracer.match import Corpus, day_window, find_matches, origin, parse_time

ROLES = ("appearances", "evidence", "counter", "hedged", "stated_as_fact", "action", "correction", "confirmation")
NOT_ORIGIN = ("correction", "confirmation")
TOLERANCE = timedelta(minutes=10)


def labelled(case):
    """Map each labelled id to its first role."""
    ids = {}
    for role in ROLES:
        for i in case.get(role, []):
            ids.setdefault(i, role)
    return ids


def check(key_path, root, scope=SCOPE, quiet=False, corpus=None, labels=None):
    """Print how each case in the key does. Returns (cases passed, cases).

    labels, from tracer.judge.load_labels, replace the rules' calls where they exist.
    """
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    corpus = corpus or Corpus(root / key["records"])
    complete = key.get("complete", False)
    by_id = {r["id"][:8]: r for r in corpus.records}
    hedges = load_hedges()
    passed = 0
    say = (lambda *a: None) if quiet else print
    if not complete:
        say(f"\n{Path(key_path).name} lists key records only, so lines marked 'note' are for review and don't fail a case.")

    for case in key["cases"]:
        since, until = day_window(corpus, case["day"]) if "day" in case else (None, None)
        matches = find_matches(case["claim"], corpus, since, until)
        judged = apply_labels(case["claim"], matches, labels) if labels else 0
        analysis = harden(matches, case["claim"], hedges, scope)
        found = {m["record"]["id"][:8]: m for m in matches}
        want = labelled(case)
        problems, notes = [], []
        strict = problems if complete else notes

        say(f"\n{case['name']}: \"{case['claim']}\"" + (f" (day {case['day']})" if "day" in case else ""))
        say(f"  found {sum(i in found for i in want)} of {len(want)} labelled records; {len(matches)} matches "
            f"({sum(m['core'] for m in matches)} core, {sum(m['strong'] for m in matches)} strong)"
            + (f", {judged} labelled by the judge" if labels else ""))
        for i in want:
            if i not in found:
                strict.append(f"MISSED {i} ({want[i]}): {one_line(by_id[i]['text'])}")
        for i in case.get("must_not_match", []):
            if i in found:
                problems.append(f"WRONGLY MATCHED {i}: {one_line(by_id[i]['text'])}")

        first = origin(matches)
        candidates = [i for i, role in want.items() if role not in NOT_ORIGIN]
        expected = min(candidates, key=lambda i: parse_time(by_id[i]["time"])) if candidates else None
        if expected and (first is None or first["record"]["id"][:8] != expected):
            strict.append(f"ORIGIN is {short(first)}, labelled {expected} ({by_id[expected]['time'][11:16]})")

        for role, stance in (("hedged", "hedged"), ("stated_as_fact", "fact")):
            for i in case.get(role, []):
                if i in found and found[i]["stance"] != stance:
                    m = found[i]
                    strict.append(f"{i} labelled {stance}, marked {m['stance']} {m['hedges'] or ''}: {one_line(m['excerpt'])}")

        point = analysis["hardening_point"]
        expected_point = case.get("hardening_point") if case.get("expect", "").startswith("hardening") else None
        say(f"  hardening point: {short(point)}, labelled {expected_point or 'none'}; "
            f"{len(analysis['hedged_before'])} hedged before it, {len(analysis['actions'])} actions after it")
        if expected_point is None:
            if point:
                problems.append(f"HARDENING POINT is {short(point)}, but none is expected")
        elif point is None:
            problems.append(f"NO HARDENING POINT, labelled {expected_point}")
        elif complete and point["record"]["id"][:8] != expected_point:
            problems.append(f"HARDENING POINT is {short(point)}, should be {expected_point}")
        elif not complete:
            gap = parse_time(point["record"]["time"]) - parse_time(by_id[expected_point]["time"])
            minutes = round(gap.total_seconds() / 60)
            if abs(gap) > TOLERANCE:
                problems.append(f"HARDENING POINT is {minutes:+d} min from the labelled one")
            else:
                say(f"  hardening point is {minutes:+d} min from the labelled one: {one_line(point['excerpt'])}")
        counted = {m["record"]["id"][:8] for m in analysis["actions"]}
        for i in case.get("action", []):
            if i not in counted:
                strict.append(f"ACTION {i} not counted")

        for p in problems:
            say("  " + p)
        for n in notes:
            say("  note: " + n)
        say("  PASS" if not problems else "  FAIL")
        passed += not problems
    return passed, len(key["cases"])


def short(match):
    if match is None:
        return "none"
    r = match["record"]
    return f"{r['id'][:8]} ({r['time'][11:16]} {r['agent']})"


def one_line(text, width=100):
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"
