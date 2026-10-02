"""Check the tracer against hand-labelled answer keys.

An answer key names a records file and, for each claim, the record ids the
tracer should find (grouped by role) and any it must not. There are two:
demo_data/expected.json labels every demo record, and test_cases.json holds the
real cases found by reading the chat.
"""

import json
from pathlib import Path

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


def check(key_path, root):
    """Print how each case in the key does. Returns True if every case passes."""
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    corpus = Corpus(root / key["records"])
    by_id = {r["id"][:8]: r for r in corpus.records}
    passed = True

    for case in key["cases"]:
        matches = find_matches(case["claim"], corpus)
        found = {m["record"]["id"][:8] for m in matches}
        want = labelled(case)
        missed = [i for i in want if i not in found]
        wrong = [i for i in case.get("must_not_match", []) if i in found]
        strong = sum(m["strong"] for m in matches)

        print(f"\n{case['name']}: \"{case['claim']}\"")
        print(f"  found {len(want) - len(missed)} of {len(want)} labelled records "
              f"({len(matches)} matches in total, {strong} strong)")
        for i in missed:
            print(f"  MISSED {i} ({want[i]}): {one_line(by_id[i]['text'])}")
        for i in wrong:
            print(f"  WRONGLY MATCHED {i}: {one_line(by_id[i]['text'])}")

        first = origin(matches)
        candidates = [i for i, role in want.items() if role not in NOT_ORIGIN]
        expected = min(candidates, key=lambda i: parse_time(by_id[i]["time"])) if candidates else None
        if first:
            r = first["record"]
            print(f"  origin: {r['id'][:8]} {r['time'][11:16]} {r['agent']}: {one_line(first['excerpt'])}")
        origin_ok = expected is None or (first is not None and first["record"]["id"][:8] == expected)
        if not origin_ok:
            print(f"  ORIGIN should be {expected}: {one_line(by_id[expected]['text'])}")

        ok = not missed and not wrong and origin_ok
        passed &= ok
        print("  PASS" if ok else "  FAIL")
    return passed


def one_line(text, width=100):
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"
