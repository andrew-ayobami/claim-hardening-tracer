"""Claim hardening tracer.

Usage:
    python trace.py index [RECORDS]          embed the records once and save the vectors
    python trace.py trace "claim" [--records RECORDS]
                                             trace a claim and find where it hardened into fact
    python trace.py check [KEY ...]          score the tracer against hand-labelled answer keys

RECORDS defaults to data/records.jsonl, which the AI Village loader writes.
Use demo_data/records.jsonl to try it on the demo dataset. KEY defaults to
demo_data/expected.json; test_cases.json holds the real cases.

--scope sets how much text around a match is checked for hedge words:
sentence, neighbours (one sentence either side, the default) or line.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDS = ROOT / "data" / "records.jsonl"
DEFAULT_KEY = ROOT / "demo_data" / "expected.json"
SCOPES = ("sentence", "neighbours", "line")


def print_trace(claim, records_path, scope):
    from tracer.evaluate import one_line
    from tracer.harden import harden
    from tracer.match import Corpus, find_matches

    matches = find_matches(claim, Corpus(records_path))
    if not matches:
        print(f'No appearances of "{claim}".')
        return
    result = harden(matches, scope=scope)
    point = result["hardening_point"]

    print(f'{len(matches)} appearances of "{claim}" ({sum(m["strong"] for m in matches)} strong)')
    print("* strong match  K keyword  F fuzzy  E embedding  ? hedged  ! hardening point\n")
    for m in matches:
        r = m["record"]
        how = "".join(name[0].upper() for name in m["methods"])
        mark = "!" if m is point else "?" if m["stance"] == "hedged" else " "
        print(f"{'*' if m['strong'] else ' '}{mark} {r['time'][:16].replace('T', ' ')}  {r['id'][:8]}  "
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
        print("\nNo hardening point: no strong match stated as fact after a hedged one.")

    from tracer.report import write_report

    source = Path(records_path).resolve()
    source = source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else source.name
    print(f"\nReport: {write_report(claim, matches, result, source, scope)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="embed the records once and save the vectors next to them")
    index.add_argument("records", nargs="?", type=Path, default=DEFAULT_RECORDS)

    trace = commands.add_parser("trace", help="trace a claim and find where it hardened into fact")
    trace.add_argument("claim")
    trace.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    trace.add_argument("--scope", choices=SCOPES, default="neighbours")

    check = commands.add_parser("check", help="score the tracer against hand-labelled answer keys")
    check.add_argument("keys", nargs="*", type=Path, default=[DEFAULT_KEY])
    check.add_argument("--scope", choices=SCOPES, default="neighbours")

    args = parser.parse_args()
    if args.command == "index":
        from tracer.index import build_index

        build_index(args.records)
    elif args.command == "trace":
        print_trace(args.claim, args.records, args.scope)
    elif args.command == "check":
        from tracer.evaluate import check as run_check

        results = [run_check(key, ROOT, args.scope) for key in args.keys]
        passed, total = sum(p for p, _ in results), sum(t for _, t in results)
        print(f"\n{passed} of {total} cases passed")
        sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
