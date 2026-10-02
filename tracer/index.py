"""Index step: embed every record's text once and save the vectors to disk.

The model only reads about the first 256 word pieces of a text, and most memory
records are longer than that. So each record is split into chunks of whole
sentences, up to about MAX_CHARS characters each, and every chunk is embedded.
A record matches a claim if any of its chunks does, and the chunk's character
span says where in the record the match is.

The index is saved next to the records file: records.jsonl -> records.index.npz.
"""

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_CHARS = 300
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
HAS_WORD = re.compile(r"\w")


def load_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_spans(text):
    """Split text into (start, end) spans of whole sentences, packed up to MAX_CHARS.

    A sentence longer than MAX_CHARS stays whole in its own chunk.
    """
    pieces, pos = [], 0
    for m in SENTENCE_END.finditer(text):
        pieces.append((pos, m.start()))
        pos = m.end()
    pieces.append((pos, len(text)))

    spans = []
    for start, end in pieces:
        if not HAS_WORD.search(text, start, end):
            continue
        if spans and end - spans[-1][0] <= MAX_CHARS:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return spans


def index_path(records_path):
    return Path(records_path).with_suffix(".index.npz")


def file_hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_index(records_path, batch_size=64):
    from sentence_transformers import SentenceTransformer  # slow to import, so only here

    started = time.time()
    records = load_records(records_path)
    record, start, end, texts = [], [], [], []
    for i, r in enumerate(records):
        for s, e in chunk_spans(r["text"]):
            record.append(i)
            start.append(s)
            end.append(e)
            texts.append(r["text"][s:e])

    # The same sentence often appears in many records, so each one is embedded once.
    unique = list(dict.fromkeys(texts))
    print(f"{len(records):,} records -> {len(texts):,} chunks ({len(unique):,} unique). Embedding with {MODEL}...")
    model = SentenceTransformer(MODEL)
    vectors = model.encode(unique, batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True)
    slot = {t: k for k, t in enumerate(unique)}

    out = index_path(records_path)
    meta = {
        "model": MODEL,
        "max_chars": MAX_CHARS,
        "records_file": Path(records_path).name,
        "records_sha1": file_hash(records_path),
        "records": len(records),
        "chunks": len(texts),
    }
    np.savez(
        out,
        vectors=vectors.astype(np.float16),
        chunk_vector=np.array([slot[t] for t in texts], dtype=np.int32),
        record=np.array(record, dtype=np.int32),
        start=np.array(start, dtype=np.int32),
        end=np.array(end, dtype=np.int32),
        meta=np.array(json.dumps(meta)),
    )
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - started:.0f} s")
    return out


def load_index(records_path):
    """Load the index for a records file, refusing one built from different records."""
    path = index_path(records_path)
    if not path.exists():
        raise SystemExit(f"No index for {records_path}. Run: python trace.py index {records_path}")
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    if meta["records_sha1"] != file_hash(records_path):
        raise SystemExit(f"{path.name} was built from a different version of {records_path}. "
                         f"Run: python trace.py index {records_path}")
    return {
        "meta": meta,
        "vectors": data["vectors"].astype(np.float32),
        "chunk_vector": data["chunk_vector"],
        "record": data["record"],
        "start": data["start"],
        "end": data["end"],
    }
