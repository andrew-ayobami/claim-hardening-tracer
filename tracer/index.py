"""Index step: embed every record's text once and save the vectors to disk.

The model only reads about the first 256 word pieces of a text, most memory
records are longer than that, and a claim is usually one sentence. So each
record is split into sentences (or lines), and every sentence is embedded.
A record matches a claim if any of its sentences does, and the sentence's
character span says where in the record the match is.

Embedding takes hours on a slow laptop, so progress is saved in blocks and a
stopped run carries on where it left off.

The index is saved next to the records file: records.jsonl -> records.index.npz.
"""

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_CHARS = 25    # shorter pieces ("Nice work, Cedar.") join a neighbouring sentence
MAX_CHARS = 300   # ...unless the joined chunk would be longer than this
BLOCK = 4096      # chunks embedded between checkpoints
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
HAS_WORD = re.compile(r"\w")


def load_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_spans(text):
    """Split text into (start, end) spans of single sentences or lines.

    A piece shorter than MIN_CHARS joins the chunk before it, and a chunk
    shorter than MIN_CHARS takes in the piece after it, as long as the result
    stays within MAX_CHARS. A sentence longer than MAX_CHARS stays whole.
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
        if spans:
            prev_start, prev_end = spans[-1]
            short = end - start < MIN_CHARS or prev_end - prev_start < MIN_CHARS
            if short and end - prev_start <= MAX_CHARS:
                spans[-1] = (prev_start, end)
                continue
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
    started = time.time()
    records = load_records(records_path)
    record, start, end, texts = [], [], [], []
    for i, r in enumerate(records):
        for s, e in chunk_spans(r["text"]):
            record.append(i)
            start.append(s)
            end.append(e)
            texts.append(r["text"][s:e])

    # The same sentence often appears in many records, so each one is embedded
    # once. Sorting by length keeps each batch's texts a similar size.
    unique = sorted(set(texts), key=lambda t: (len(t), t))
    print(f"{len(records):,} records -> {len(texts):,} chunks ({len(unique):,} unique) to embed with {MODEL}")
    vectors = embed_with_checkpoints(unique, index_path(records_path).with_suffix(".partial"), batch_size)
    slot = {t: k for k, t in enumerate(unique)}

    out = index_path(records_path)
    meta = {
        "model": MODEL,
        "min_chars": MIN_CHARS,
        "max_chars": MAX_CHARS,
        "records_file": Path(records_path).name,
        "records_sha1": file_hash(records_path),
        "records": len(records),
        "chunks": len(texts),
    }
    np.savez(
        out,
        vectors=vectors,
        chunk_vector=np.array([slot[t] for t in texts], dtype=np.int32),
        record=np.array(record, dtype=np.int32),
        start=np.array(start, dtype=np.int32),
        end=np.array(end, dtype=np.int32),
        meta=np.array(json.dumps(meta)),
    )
    shutil.rmtree(index_path(records_path).with_suffix(".partial"), ignore_errors=True)
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB) in {format_duration(time.time() - started)}")
    return out


def embed_with_checkpoints(texts, folder, batch_size):
    """Embed texts in blocks of BLOCK, saving each block to `folder` as it finishes.

    A rerun on the same texts loads the saved blocks and embeds only the rest.
    """
    from sentence_transformers import SentenceTransformer  # slow to import, so only here

    key = hashlib.sha1(json.dumps([MODEL, BLOCK, texts]).encode("utf-8")).hexdigest()
    folder.mkdir(exist_ok=True)
    manifest = folder / "manifest.json"
    if not manifest.exists() or json.loads(manifest.read_text())["key"] != key:
        for old in folder.glob("block_*.npy"):
            old.unlink()
        manifest.write_text(json.dumps({"key": key, "texts": len(texts)}))

    total_chars = sum(len(t) for t in texts)
    blocks, model, done_chars, new_chars, began = [], None, 0, 0, time.time()
    for b, first in enumerate(range(0, len(texts), BLOCK)):
        block = texts[first:first + BLOCK]
        path = folder / f"block_{b:05d}.npy"
        if path.exists():
            blocks.append(np.load(path))
            done_chars += sum(len(t) for t in block)
            continue
        if model is None:
            if b:
                print(f"Resuming: {first:,} chunks were already embedded")
            model = SentenceTransformer(MODEL)
            began = time.time()
        vectors = model.encode(block, batch_size=batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            np.save(f, vectors)
        os.replace(tmp, path)
        blocks.append(vectors)

        chars = sum(len(t) for t in block)
        done_chars += chars
        new_chars += chars
        left = (total_chars - done_chars) * (time.time() - began) / new_chars
        print(f"  {min(first + BLOCK, len(texts)):,} of {len(texts):,} embedded, "
              f"about {format_duration(left)} left", flush=True)
    return np.concatenate(blocks) if blocks else np.zeros((0, 384), dtype=np.float16)


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60} min"


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
