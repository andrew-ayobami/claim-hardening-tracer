"""Download the AI Village tables this project uses into data/raw/.

The dataset is gated. Request access on its Hugging Face page, then run
`hf auth login` once before running this script.

Downloads resume: each file is written to `<name>.part` first, so if you stop
the script (Ctrl+C) and run it again, it carries on where it left off. Dropped
connections are retried automatically from the same point. Finished
files are checked against the checksum Hugging Face publishes, then skipped on
later runs.
"""

import hashlib
import re
import time
from pathlib import Path

import httpx
from huggingface_hub import get_hf_file_metadata, get_session, hf_hub_url
from huggingface_hub.utils import build_hf_headers

REPO_ID = "aidigestorg/ai-village"
OUT_DIR = Path(__file__).parent / "data" / "raw"
CHUNK = 1024 * 1024
REPORT_EVERY = 25 * 1024 * 1024
RETRIES = 20

# Only the tables the tracer needs. The screenshot archives under images/
# make up most of the dataset and are skipped.
FILES = [
    "README.md",
    "SCHEMA.md",
    "CHANGELOG.md",
    "agents.jsonl.gz",
    "chat_rooms.jsonl.gz",
    "village_goals.jsonl.gz",
    "chat_messages.jsonl.gz",
    "computer_use_sessions.jsonl.gz",
    "village-transcript.json",  # about 345 MB; maps village day numbers to dates
    "agent_memories.jsonl.gz",  # about 2.3 GB, so it goes last
]


def download(name):
    dest = OUT_DIR / name
    if dest.exists():
        print(f"{name}: already downloaded")
        return

    url = hf_hub_url(REPO_ID, name, repo_type="dataset")
    meta = get_hf_file_metadata(url)
    part = dest.with_name(dest.name + ".part")

    for attempt in range(1, RETRIES + 1):
        try:
            fetch(name, url, part, meta.size)
            break
        except httpx.TransportError as e:
            if attempt == RETRIES:
                raise
            print(f"{name}: connection dropped ({e}), resuming", flush=True)
            time.sleep(5)

    check(part, meta)
    part.replace(dest)
    print(f"{name}: done, {meta.size / 1e6:,.1f} MB")


def fetch(name, url, part, size):
    """Download the rest of `url` into `part`, starting from what's already there."""
    done = part.stat().st_size if part.exists() else 0
    if done >= size:
        return
    headers = build_hf_headers()
    if done:
        headers["Range"] = f"bytes={done}-"
    with get_session().stream("GET", url, headers=headers, follow_redirects=True) as r:
        r.raise_for_status()
        if done and r.status_code != 206:  # server ignored the range, so start over
            done = 0
        with open(part, "ab" if done else "wb") as f:
            next_report = done + REPORT_EVERY
            for chunk in r.iter_bytes(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    print(f"{name}: {done / 1e6:,.0f} of {size / 1e6:,.0f} MB", flush=True)
                    next_report += REPORT_EVERY


def check(path, meta):
    """Fail if the file doesn't match the size and checksum Hugging Face publishes."""
    size = path.stat().st_size
    if size != meta.size:
        raise RuntimeError(f"{path.name}: expected {meta.size} bytes, got {size}")
    # For large files the etag is the SHA-256 of the contents.
    if meta.etag and re.fullmatch(r"[0-9a-f]{64}", meta.etag):
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(CHUNK), b""):
                sha.update(block)
        if sha.hexdigest() != meta.etag:
            path.unlink()
            raise RuntimeError(f"{path.name}: checksum mismatch, deleted it; run again")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download(name)


if __name__ == "__main__":
    main()
