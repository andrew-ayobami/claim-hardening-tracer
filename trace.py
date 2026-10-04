"""Claim hardening tracer.

Usage:
    python trace.py index [RECORDS]          embed the records once and save the vectors
    python trace.py trace "claim" [--records RECORDS] [--day N | --since T --until T]
                                             trace a claim, find where it hardened into fact,
                                             and write reports/<claim>.html
    python trace.py check [KEY ...]          score the tracer against hand-labelled answer keys
    python trace.py export [KEY ...] [--claim "claim" --day N]
                                             write the matches for the language-model judge
                                             (kaggle/judge.py) to data/judge/candidates.jsonl

RECORDS defaults to data/records.jsonl, which the AI Village loader writes.
Use demo_data/records.jsonl to try it on the demo dataset. KEY defaults to
demo_data/expected.json; test_cases.json holds the real cases.

--day limits the search to one AI Village day; --since and --until take ISO
times in UTC (2026-02-09 or 2026-02-09T20:00). --scope sets how much text
around a match is checked for hedge words: sentence, neighbours (one sentence
either side) or line (the default). --compact writes a smaller report with only
close restatements, hedged appearances and actions, for sharing. --labels
reads the judge's answers (see tracer/judge.py) for trace and check.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDS = ROOT / "data" / "records.jsonl"
DEFAULT_KEY = ROOT / "demo_data" / "expected.json"
DEFAULT_CANDIDATES = ROOT / "data" / "judge" / "candidates.jsonl"
SCOPES = ("sentence", "neighbours", "line")
MARKS = {"hedged": "?", "disputed": "x", "question": "q", "fact": " "}


def utc_time(text):
    t = datetime.fromisoformat(text)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def print_trace(claim, records_path, scope, day=None, since=None, until=None, compact=False, labels=None):
    from tracer.evaluate import one_line
    from tracer.harden import harden
    from tracer.judge import apply_labels
    from tracer.match import Corpus, day_window, find_matches
    from tracer.report import write_report

    corpus = Corpus(records_path)
    if day is not None:
        since, until = day_window(corpus, day)
    matches = find_matches(claim, corpus, since, until)
    if not matches:
        print(f'No appearances of "{claim}".')
        return
    judged = apply_labels(claim, matches, labels) if labels else 0
    result = harden(matches, claim, scope=scope)
    point = result["hardening_point"]

    print(f'{len(matches)} appearances of "{claim}" ({sum(m["core"] for m in matches)} core, '
          f'{sum(m["strong"] for m in matches)} strong' + (f", {judged} labelled by the judge)" if labels else ")"))
    print("C core  * strong  K keyword  F fuzzy  E embedding  ? hedged  x disputed  q question  ! hardening point\n")
    for m in matches:
        r = m["record"]
        how = "".join(name[0].upper() for name in m["methods"])
        tier = "C" if m["core"] else "*" if m["strong"] else " "
        mark = "!" if m is point else MARKS[m["stance"]]
        print(f"{tier}{mark} {r['time'][:16].replace('T', ' ')}  {r['id'][:8]}  "
              f"{r['agent'][:22]:22} {r['channel'][:14]:14} {how:3} {m['similarity']:.2f}  {one_line(m['excerpt'], 90)}")

    for label, m in (("Origin", result["origin"]), ("Hardening point", point)):
        if m:
            r = m["record"]
            print(f"\n{label}: {r['id'][:8]} at {r['time'][:16].replace('T', ' ')} by {r['agent']} ({r['channel']})\n  {r['link']}")
    if point:
        hedges = sorted({h for m in result["hedged_before"] for h in m["hedges"]}, key=str.lower)
        print(f"  after {len(result['hedged_before'])} hedged appearances ({', '.join(hedges)}), "
              f"then {len(result['actions'])} computer-use sessions acted on it")
    else:
        print("\nNo hardening point: no close restatement was stated as fact after a hedged appearance.")

    source = Path(records_path).resolve()
    source = source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else source.name
    if day is not None:
        source += f", day {day}"
    print(f"\nReport: {write_report(claim, matches, result, source, scope, compact=compact)}")


def export_candidates(keys, out, claim=None, records_path=DEFAULT_RECORDS, day=None, since=None, until=None):
    import json

    from tracer.judge import export
    from tracer.match import Corpus, day_window, find_matches

    traces, corpora = [], {}

    def add(claim, path, day, since, until):
        corpus = corpora.setdefault(path, Corpus(path))
        if day is not None:
            since, until = day_window(corpus, day)
        matches = find_matches(claim, corpus, since, until)
        print(f"{len(matches):5} matches  {claim}")
        traces.append((claim, matches))

    if claim:
        add(claim, records_path, day, since, until)
    for key_path in keys:
        key = json.loads(Path(key_path).read_text(encoding="utf-8"))
        for case in key["cases"]:
            add(case["claim"], ROOT / key["records"], case.get("day"), None, None)
    print(f"\n{export(traces, out)} candidates written to {out}")


def main():
    from tracer.harden import SCOPE

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="embed the records once and save the vectors next to them")
    index.add_argument("records", nargs="?", type=Path, default=DEFAULT_RECORDS)

    trace = commands.add_parser("trace", help="trace a claim and find where it hardened into fact")
    trace.add_argument("claim")
    trace.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    trace.add_argument("--day", type=int, help="limit the search to one AI Village day")
    trace.add_argument("--since", type=utc_time, help="limit the search to records from this UTC time")
    trace.add_argument("--until", type=utc_time, help="...and before this UTC time")
    trace.add_argument("--scope", choices=SCOPES, default=SCOPE)
    trace.add_argument("--compact", action="store_true", help="smaller report for sharing")
    trace.add_argument("--labels", type=Path, help="the language-model judge's answers (labels.jsonl)")

    check = commands.add_parser("check", help="score the tracer against hand-labelled answer keys")
    check.add_argument("keys", nargs="*", type=Path, default=[DEFAULT_KEY])
    check.add_argument("--scope", choices=SCOPES, default=SCOPE)
    check.add_argument("--labels", type=Path, help="the language-model judge's answers (labels.jsonl)")

    export = commands.add_parser("export", help="write the matches for the language-model judge")
    export.add_argument("keys", nargs="*", type=Path, help="answer keys whose cases to export")
    export.add_argument("--claim", help="a claim to export instead of, or as well as, the keys' cases")
    export.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    export.add_argument("--day", type=int)
    export.add_argument("--since", type=utc_time)
    export.add_argument("--until", type=utc_time)
    export.add_argument("--out", type=Path, default=DEFAULT_CANDIDATES)

    args = parser.parse_args()
    labels = None
    if getattr(args, "labels", None):
        from tracer.judge import load_labels

        labels = load_labels(args.labels)
    if args.command == "index":
        from tracer.index import build_index

        build_index(args.records)
    elif args.command == "trace":
        print_trace(args.claim, args.records, args.scope, args.day, args.since, args.until, args.compact, labels)
    elif args.command == "check":
        from tracer.evaluate import check as run_check

        results = [run_check(key, ROOT, args.scope, labels=labels) for key in args.keys]
        passed, total = sum(p for p, _ in results), sum(t for _, t in results)
        print(f"\n{passed} of {total} cases passed")
        sys.exit(0 if passed == total else 1)
    elif args.command == "export":
        if not args.keys and not args.claim:
            parser.error("export needs answer keys or --claim")
        export_candidates(args.keys, args.out, args.claim, args.records, args.day, args.since, args.until)


if __name__ == "__main__":
    main()
