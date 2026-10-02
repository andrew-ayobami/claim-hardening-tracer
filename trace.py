"""Claim hardening tracer.

Usage:
    python trace.py index [RECORDS]   embed the records once and save the vectors

RECORDS defaults to data/records.jsonl, which the AI Village loader writes.
Use demo_data/records.jsonl to try it on the demo dataset.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDS = ROOT / "data" / "records.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="embed the records once and save the vectors next to them")
    index.add_argument("records", nargs="?", type=Path, default=DEFAULT_RECORDS)
    args = parser.parse_args()

    if args.command == "index":
        from tracer.index import build_index

        build_index(args.records)


if __name__ == "__main__":
    main()
