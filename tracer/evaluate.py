"""Check the tracer against hand-labelled answer keys.

An answer key names a records file and, for each claim, the record ids the
tracer should find (grouped by role) and any it must not. There are two:
demo_data/expected.json labels every demo record, and test_cases.json holds the
real cases found by reading the chat.

For each case it checks the lineage (every labelled record found, the near
misses left out, the right origin) and the hardening (the labelled hedged and
fact records marked that way, the right hardening point, the actions counted).
"""

import json
from pathlib import Path

from tracer.harden import SCOPE, harden, load_hedges
from tracer.match import Corpus, find_matches, origin, parse_time

ROLES = ("appearances", "evidence", "counter", "hedged", "stated_as_fact", "action", "correction", "confirmation")
NOT_ORIGIN = ("correction", "confirmation")


def labelled(case):
    """Map each labelled id to its first role."""
    ids = {}
    for role in ROLES:
        for i in case.get(role, []):
            ids.setdefault(i, role)
    return ids


def check(key_path, root, scope=SCOPE, quiet=False):
    """Print how each case in the key does. Returns (cases passed, cases)."""
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    corpus = Corpus(root / key["records"])
    by_id = {r["id"][:8]: r for r in corpus.records}
    hedges = load_hedges()
    passed = 0
    say = (lambda *a: None) if quiet else print

    for case in key["cases"]:
        matches = find_matches(case["claim"], corpus)
        analysis = harden(matches, hedges, scope)
        found = {m["record"]["id"][:8]: m for m in matches}
        want = labelled(case)
        problems = []

        say(f"\n{case['name']}: \"{case['claim']}\"")
        say(f"  found {sum(i in found for i in want)} of {len(want)} labelled records "
            f"({len(matches)} matches in total, {sum(m['strong'] for m in matches)} strong)")
        for i in want:
            if i not in found:
                problems.append(f"MISSED {i} ({want[i]}): {one_line(by_id[i]['text'])}")
        for i in case.get("must_not_match", []):
            if i in found:
                problems.append(f"WRONGLY MATCHED {i}: {one_line(by_id[i]['text'])}")

        first = origin(matches)
        candidates = [i for i, role in want.items() if role not in NOT_ORIGIN]
        expected = min(candidates, key=lambda i: parse_time(by_id[i]["time"])) if candidates else None
        if expected and (first is None or first["record"]["id"][:8] != expected):
            problems.append(f"ORIGIN is {short(first)}, should be {expected}")

        for role, stance in (("hedged", "hedged"), ("stated_as_fact", "fact")):
            for i in case.get(role, []):
                if i in found and found[i]["stance"] != stance:
                    m = found[i]
                    problems.append(f"{i} should be {stance}, marked {m['stance']} {m['hedges'] or ''}: "
                                    f"{one_line(m['excerpt'])}")

        point = analysis["hardening_point"]
        expected_point = case.get("hardening_point") if case.get("expect", "").startswith("hardening") else None
        say(f"  hardening point: {short(point)} (expected {expected_point or 'none'}), "
            f"after {len(analysis['hedged_before'])} hedged; {len(analysis['actions'])} actions after it")
        if (point["record"]["id"][:8] if point else None) != expected_point:
            problems.append(f"HARDENING POINT is {short(point)}, should be {expected_point or 'none'}")
        counted = {m["record"]["id"][:8] for m in analysis["actions"]}
        for i in case.get("action", []):
            if i not in counted:
                problems.append(f"ACTION {i} not counted")

        for p in problems:
            say("  " + p)
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
