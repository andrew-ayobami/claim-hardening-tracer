"""Claim hardening tracer.

Usage:
    python trace.py index [RECORDS]          embed the records once and save the vectors
    python trace.py trace "claim" [--records RECORDS]
                                             list every appearance of a claim in time order
    python trace.py check [KEY ...]          score the tracer against hand-labelled answer keys

RECORDS defaults to data/records.jsonl, which the AI Village loader writes.
Use demo_data/records.jsonl to try it on the demo dataset. KEY defaults to
demo_data/expected.json; test_cases.json holds the real cases.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDS = ROOT / "data" / "records.jsonl"
DEFAULT_KEY = ROOT / "demo_data" / "expected.json"


def print_trace(claim, records_path):
    from tracer.evaluate import one_line
    from tracer.match import Corpus, find_matches, origin

    corpus = Corpus(records_path)
    matches = find_matches(claim, corpus)
    if not matches:
        print(f'No appearances of "{claim}".')
        return
    print(f'{len(matches)} appearances of "{claim}" ({sum(m["strong"] for m in matches)} strong, '
          f"* marks strong; K keyword, F fuzzy, E embedding)\n")
    for m in matches:
        r = m["record"]
        how = "".join(name[0].upper() for name in m["methods"])
        print(f"{'*' if m['strong'] else ' '} {r['time'][:16].replace('T', ' ')}  {r['id'][:8]}  "
              f"{r['agent'][:22]:22} {r['channel'][:14]:14} {how:3} {m['similarity']:.2f}  {one_line(m['excerpt'], 90)}")
    first = origin(matches)
    if first:
        r = first["record"]
        print(f"\nOrigin: {r['id'][:8]} at {r['time'][:16].replace('T', ' ')} by {r['agent']} ({r['channel']})\n{r['link']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="embed the records once and save the vectors next to them")
    index.add_argument("records", nargs="?", type=Path, default=DEFAULT_RECORDS)

    trace = commands.add_parser("trace", help="list every appearance of a claim in time order")
    trace.add_argument("claim")
    trace.add_argument("--records", type=Path, default=DEFAULT_RECORDS)

    check = commands.add_parser("check", help="score the tracer against hand-labelled answer keys")
    check.add_argument("keys", nargs="*", type=Path, default=[DEFAULT_KEY])

    args = parser.parse_args()
    if args.command == "index":
        from tracer.index import build_index

        build_index(args.records)
    elif args.command == "trace":
        print_trace(args.claim, args.records)
    elif args.command == "check":
        from tracer.evaluate import check as run_check

        results = [run_check(key, ROOT) for key in args.keys]
        sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
