"""Trace step: find every appearance of a claim, three ways, and sort them by time.

A chunk of a record matches the claim if any of these fire:
- keyword: at least KEYWORD_MIN of the claim's content words appear in the chunk
  (words are lowercased and lightly stemmed, so "submitted" matches "submit")
- fuzzy: the chunk contains a near-copy of the claim (rapidfuzz, FUZZY_MIN or more)
- embedding: cosine similarity of EMBED_MIN or more to the expanded claim

Claims change wording as they spread ("submitted" becomes "went through"). So
the embedding search runs twice: first with the claim alone, then with the
claim plus the average of the chunks that matched strongly the first time,
which adds the wording the agents actually used.

A record matches if any of its chunks do. A match is strong if keyword or fuzzy
fired, or the similarity is at least EMBED_STRONG; otherwise it's weak. Weak
matches are often same-topic chatter, so they stay in the lineage but can't be
the origin. Matches are sorted by time, and the earliest strong one is the origin.

The thresholds were set on the demo dataset and checked against test_cases.json.
"""

import re
from datetime import datetime

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

from tracer.index import MODEL, load_index, load_records

KEYWORD_MIN = 0.6
FUZZY_MIN = 80
SEED_MIN = 0.6      # first-pass similarity that counts as a strong match for expanding the claim
EMBED_MIN = 0.54    # second-pass similarity for a match
EMBED_STRONG = 0.65  # second-pass similarity for a strong match

STOP_WORDS = frozenset("""
a an the of to for in on at by with from into about and or but nor so yet if then than as
is are was were be been being am it its this that these those there here
i we you they he she me us them my our your their his her
do does did done has have had having will would can could should shall may might must
not no just also all any some each every very too more most such only own same
""".split())


def stem(word):
    """Strip a common ending, then a doubled final consonant: submitted -> submit."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) >= 4 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]
    return word


def content_words(text):
    return {stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP_WORDS}


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


class Corpus:
    """Records plus their index, loaded once and reused across claims."""

    def __init__(self, records_path):
        self.path = records_path
        self.records = load_records(records_path)
        index = load_index(records_path)
        self.meta = index["meta"]
        self.vectors = index["vectors"]
        self.chunk_vector = index["chunk_vector"]
        self.chunk_record = index["record"]
        self.chunk_start = index["start"]
        self.chunk_end = index["end"]
        self.chunk_text = [self.records[r]["text"][s:e]
                           for r, s, e in zip(self.chunk_record, self.chunk_start, self.chunk_end)]
        self._chunk_words = None
        self._model = None

    @property
    def chunk_words(self):
        if self._chunk_words is None:
            self._chunk_words = [content_words(t) for t in self.chunk_text]
        return self._chunk_words

    def embed(self, text):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # slow to import, so only here

            self._model = SentenceTransformer(MODEL)
        return self._model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]


def chunk_scores(claim, corpus, keyword_min=KEYWORD_MIN, fuzzy_min=FUZZY_MIN):
    """Return per-chunk keyword share, fuzzy score and similarity to the expanded claim."""
    words = content_words(claim)
    if words:
        need = len(words) if len(words) == 1 else 2
        overlap = np.array([len(words & w) for w in corpus.chunk_words])
        keyword = np.where(overlap >= need, overlap / len(words), 0.0)
    else:
        keyword = np.zeros(len(corpus.chunk_text))

    # Near-copies: look for the claim inside chunks at least as long as it, and
    # compare whole texts for shorter chunks, which would otherwise score 100
    # just for being a piece of the claim.
    partial = process.cdist([claim], corpus.chunk_text, scorer=fuzz.partial_ratio,
                            processor=default_process, workers=-1)[0]
    whole = process.cdist([claim], corpus.chunk_text, scorer=fuzz.ratio,
                          processor=default_process, workers=-1)[0]
    longer = np.array([len(t) >= len(claim) for t in corpus.chunk_text])
    fuzzy = np.where(longer, partial, whole)

    query = corpus.embed(claim)
    similarity = (corpus.vectors @ query)[corpus.chunk_vector]
    seeds = (keyword >= keyword_min) | (fuzzy >= fuzzy_min) | (similarity >= SEED_MIN)
    if seeds.any():
        expanded = query + corpus.vectors[corpus.chunk_vector[seeds]].mean(axis=0)
        similarity = (corpus.vectors @ (expanded / np.linalg.norm(expanded)))[corpus.chunk_vector]
    return keyword, fuzzy, similarity


def find_matches(claim, corpus, keyword_min=KEYWORD_MIN, fuzzy_min=FUZZY_MIN,
                 embed_min=EMBED_MIN, embed_strong=EMBED_STRONG):
    """Return every record that matches the claim, sorted by time.

    Each match keeps its best chunk: the excerpt, its span in the record text,
    the methods that fired on it and their scores.
    """
    keyword, fuzzy, embedding = chunk_scores(claim, corpus, keyword_min, fuzzy_min)
    fired = (keyword >= keyword_min) | (fuzzy >= fuzzy_min) | (embedding >= embed_min)

    best = {}
    for c in np.flatnonzero(fired):
        rec = int(corpus.chunk_record[c])
        methods = [name for name, hit in (("keyword", keyword[c] >= keyword_min),
                                          ("fuzzy", fuzzy[c] >= fuzzy_min),
                                          ("embedding", embedding[c] >= embed_min)) if hit]
        strong = methods != ["embedding"] or embedding[c] >= embed_strong
        rank = (strong, len(methods), float(embedding[c]))
        if rec not in best or rank > best[rec]["rank"]:
            best[rec] = {
                "rank": rank,
                "record": corpus.records[rec],
                "strong": bool(strong),
                "methods": methods,
                "keyword": round(float(keyword[c]), 2),
                "fuzzy": round(float(fuzzy[c])),
                "similarity": round(float(embedding[c]), 3),
                "span": (int(corpus.chunk_start[c]), int(corpus.chunk_end[c])),
                "excerpt": corpus.chunk_text[c],
            }

    matches = sorted(best.values(), key=lambda m: (parse_time(m["record"]["time"]), m["record"]["id"]))
    for m in matches:
        del m["rank"]
    return matches


def origin(matches):
    """The earliest strong match, or None."""
    return next((m for m in matches if m["strong"]), None)
