"""Load one AI Village goal period into a flat, time-sorted list of records.

Usage:
    python -m loaders.ai_village --goal "Adopt a park"

Reads the tables that download_data.py put in data/raw/ and writes:
- data/records.jsonl: one record per chat message, memory update and
  computer-use session, each with id, time, agent, channel, text and link
- data/chat.txt: the period's chat as plain text, for finding test cases by hand

Each memory row in the dataset is a full snapshot of an agent's memory, saved
many times a day. A memory record holds only the lines that snapshot added, so
it marks the moment a claim entered the agent's memory.
"""

import argparse
import bisect
import gzip
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SITE = "https://theaidigest.org/village"

# A memory row is often 30,000 characters, so lines are filtered on their
# timestamp before being decoded. Quotes inside JSON strings are escaped,
# so this can only match the real created_at key.
CREATED_AT = re.compile(r'"created_at"\s*:\s*"([^"]+)"')
AGENT_ID = re.compile(r'"agent_id"\s*:\s*"([^"]+)"')
# A chat message is saved a moment before its transcript event, so a day's
# first message can fall just before that day's recorded start.
DAY_MARGIN = timedelta(minutes=10)
# The start of each day in village-transcript.json, with its first event.
DAY_START = re.compile(
    r'\{\s*"day"\s*:\s*(\d+)\s*,\s*"date"\s*:\s*"[^"]+"\s*,'
    r'\s*"events"\s*:\s*\[\s*\{\s*"timestamp"\s*:\s*"([^"]+)"'
)


def parse_time(text):
    """Dataset timestamps are UTC with no timezone suffix."""
    t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def read_rows(name, start=None, end=None):
    """Yield the rows of a .jsonl.gz table, optionally only those created in [start, end)."""
    with gzip.open(RAW / name, "rt", encoding="utf-8") as f:
        for line in f:
            if start is not None:
                m = CREATED_AT.search(line)
                if not m or not start <= parse_time(m.group(1)) < end:
                    continue
            yield json.loads(line)


def find_goal(text):
    """Return (goal, start, end) for the one village goal whose text contains `text`."""
    goals = list(read_rows("village_goals.jsonl.gz"))
    matches = [g for g in goals if text.lower() in g["goal"].lower()]
    if len(matches) != 1:
        found = "\n".join(f"  {g['start_time'][:10]}  {g['goal']}" for g in matches or goals)
        raise SystemExit(f'"{text}" matched {len(matches)} goals. Pick one of:\n{found}')
    g = matches[0]
    end = parse_time(g["end_time"]) if g["end_time"] else datetime.now(timezone.utc)
    return g["goal"], parse_time(g["start_time"]), end


def load_day_starts():
    """Return sorted (start time, day number) pairs from village-transcript.json.

    Day numbers skip most weekends, so they can't be counted from the date.
    The file is 345 MB, so it's scanned in chunks rather than parsed whole.
    """
    starts = {}
    tail = ""
    with open(RAW / "village-transcript.json", encoding="utf-8") as f:
        while chunk := f.read(16 * 1024 * 1024):
            text = tail + chunk
            for m in DAY_START.finditer(text):
                starts[int(m.group(1))] = parse_time(m.group(2)) - DAY_MARGIN
            tail = text[-500:]
    return sorted((t, day) for day, t in starts.items())


def memory_updates(start, end):
    """Yield (row, added lines) for each memory snapshot saved in [start, end).

    The file isn't sorted by time, so it's read in full. Each agent's last
    snapshot before the period is kept too, so its first update in the period
    shows only what changed.
    """
    rows, before = [], {}
    with gzip.open(RAW / "agent_memories.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            t = parse_time(CREATED_AT.search(line).group(1))
            if start <= t < end:
                rows.append(json.loads(line))
            elif t < start:
                agent = AGENT_ID.search(line).group(1)
                if agent not in before or t > before[agent][0]:
                    before[agent] = (t, line)

    def lines(text):
        return [s for s in (raw.strip() for raw in (text or "").splitlines()) if s]

    seen = {agent: set(lines(json.loads(line)["content"])) for agent, (_, line) in before.items()}
    for row in sorted(rows, key=lambda r: parse_time(r["created_at"])):
        current = lines(row["content"])
        added = [s for s in current if s not in seen.get(row["agent_id"], set())]
        seen[row["agent_id"]] = set(current)
        if added:
            yield row, "\n".join(added)


def make_link(t, day_starts):
    i = bisect.bisect_right(day_starts, (t, float("inf"))) - 1
    day = day_starts[max(i, 0)][1]
    return f"{SITE}?day={day}&time={int(t.timestamp() * 1000)}"


def load(goal_text):
    goal, start, end = find_goal(goal_text)
    print(f"Goal: {goal}\nPeriod: {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC")

    agents = {a["id"]: a["name"] for a in read_rows("agents.jsonl.gz")}
    rooms = {r["id"]: r["name"] for r in read_rows("chat_rooms.jsonl.gz")}
    day_starts = load_day_starts()
    records = []

    def add(row_id, created_at, agent, channel, text):
        t = parse_time(created_at)
        records.append({
            "id": row_id,
            "time": t.isoformat().replace("+00:00", "Z"),
            "agent": agent,
            "channel": channel,
            "text": text or "",
            "link": make_link(t, day_starts),
        })

    for m in read_rows("chat_messages.jsonl.gz", start, end):
        # Human speakers have no names in the dataset.
        agent = agents.get(m["agent_speaker_id"], m["agent_speaker_id"]) if m["speaker_type"] == "agent" else "human"
        add(m["id"], m["created_at"], agent, f"chat:{rooms.get(m['room_id'], m['room_id'])}", m["content"])

    for s in read_rows("computer_use_sessions.jsonl.gz", start, end):
        add(s["id"], s["created_at"], agents.get(s["agent_id"], s["agent_id"]), "computer use", s["session_goal"])

    print("Scanning agent memories (2.4 GB). This takes a few minutes...")
    for m, added in memory_updates(start, end):
        add(m["id"], m["created_at"], agents.get(m["agent_id"], m["agent_id"]), "memory", added)

    records.sort(key=lambda r: (r["time"], r["id"]))
    return goal, start, end, records


def write_records(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_chat(goal, start, end, records, path):
    """Write the chat as plain text, grouped by village day, with short record ids."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Chat for: {goal}\n{start:%Y-%m-%d} to {end:%Y-%m-%d}. Times are UTC.\n")
        f.write("Each message starts with the first 8 characters of its record id.\n")
        day = None
        for r in records:
            if not r["channel"].startswith("chat:"):
                continue
            this_day = re.search(r"day=(\d+)", r["link"]).group(1)
            if this_day != day:
                day = this_day
                f.write(f"\n\n===== Day {day}, {parse_time(r['time']):%a %d %b %Y} =====\n")
            room = r["channel"].removeprefix("chat:")
            f.write(f"\n[{r['id'][:8]}] {r['time'][11:16]} {r['agent']} (#{room})\n{r['text'].strip()}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--goal", required=True, help="words from the village goal, e.g. 'Adopt a park'")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "records.jsonl")
    parser.add_argument("--chat", type=Path, default=ROOT / "data" / "chat.txt")
    args = parser.parse_args()

    goal, start, end, records = load(args.goal)
    write_records(records, args.out)
    write_chat(goal, start, end, records, args.chat)

    counts = {}
    for r in records:
        kind = "chat" if r["channel"].startswith("chat:") else r["channel"]
        counts[kind] = counts.get(kind, 0) + 1
    print(f"Wrote {len(records):,} records to {args.out}: " + ", ".join(f"{n:,} {k}" for k, n in counts.items()))
    print(f"Wrote the readable chat to {args.chat}")


if __name__ == "__main__":
    main()
