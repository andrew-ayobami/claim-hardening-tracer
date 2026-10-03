"""Trace step: find every appearance of a claim, three ways, and sort them by time.

A chunk (sentence) of a record matches the claim if any of these fire:
- keyword: the chunk holds at least KEYWORD_MIN of the claim's content words,
  weighted by rarity, so "glitch" counts for more than "agents" or "GitHub"
  (words are lowercased and lightly stemmed, so "submitted" matches "submit")
- fuzzy: the chunk contains a near-copy of the claim (rapidfuzz, FUZZY_MIN or more)
- embedding: cosine similarity of EMBED_MIN or more to the expanded claim

Claims change wording as they spread ("submitted" becomes "went through"). So
the embedding search runs twice: first with the claim alone, then with the
claim plus the average of its SEEDS closest strong matches, which adds the
wording the agents actually used. Averaging only the closest few keeps the
claim from drifting towards the general topic.

Each match gets a tier:
- core: a close restatement of the claim: similarity of CORE_MIN or more (or a
  near-copy), and most of the claim's lowercase words, the parts that aren't
  names. A short claim like "The Love Dolores outreach is working" is mostly
  names, so similarity alone lets in any sentence about the outreach; this
  keeps "working" in. Hardening is judged on these.
- strong: keyword or fuzzy fired, or similarity of EMBED_STRONG or more. The
  earliest strong match is the origin.
- weak: embedding only, below EMBED_STRONG. Often same-topic chatter, so it
  stays in the lineage but plays no part in the analysis.

Numbers are part of what a claim says ("5 minutes left", "$115 raised"). So if
the claim has a standalone number, a sentence without it can only be weak: "~26
minutes left" is a different claim from "5 minutes left", however alike they
read. Version numbers in names, like GPT-5.2 or Gemini 2.5 Pro, don't count.

A search can be limited to a time window (since, until), for example one
village day, which is how an investigator usually starts.

The thresholds were set on the demo dataset and checked against test_cases.json.
"""

import math
import re
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

from tracer.index import MODEL, load_index, load_records

KEYWORD_MIN = 0.6    # rarity-weighted share of the claim's words
FUZZY_MIN = 80
SEED_MIN = 0.6       # first-pass similarity that makes a chunk a candidate seed
SEEDS = 10           # closest seeds averaged into the expanded claim
EMBED_MIN = 0.47     # second-pass similarity for a (weak) match: low, since weak matches don't affect the analysis
EMBED_STRONG = 0.65  # second-pass similarity for a strong match
CORE_MIN = 0.7       # second-pass similarity for a close restatement

NUMBER = re.compile(r"(?<![\w.-])\d+(?:\.\d+)?(?![\w-]|\.\d)")
NAME_VERSION = re.compile(r"\b[A-Z][A-Za-z]+\s+\d+(?:\.\d+)+\b")  # "Gemini 2.5", "Opus 4.6"

STOP_WORDS = frozenset("""
a an the of to for in on at by with from into about and or but nor so yet if then than as
is are was were be been being am it its this that these those there here
i we you they he she me us them my our your their his her
do does did done has have had having will would can could should shall may might must
not no just also all any some each every very too more most such only same
isn aren wasn weren don doesn didn hasn haven hadn won wouldn couldn shouldn t s ll ve re d m
why how what when where who whom which whose
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


def numbers(text):
    """Standalone numbers in a text, leaving out version numbers in names."""
    return set(NUMBER.findall(NAME_VERSION.sub(" ", text)))


def predicate_words(claim):
    """The claim's lowercase words: what it says, as opposed to the names it mentions."""
    return {stem(w) for w in re.findall(r"[A-Za-z]+", claim)
            if w[0].islower() and len(w) > 2 and w not in STOP_WORDS}


def says_enough(predicates, words):
    """True if more than half the claim's predicate words appear, allowing endings ("own" -> "owner")."""
    if not predicates:
        return True
    shared = sum(any(w.startswith(p) for w in words) for p in predicates)
    return shared > len(predicates) / 2


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


class Corpus:
    """Records plus their index, loaded once and reused across claims."""

    def __init__(self, records_path):
        self.path = records_path
        self.records = load_records(records_path)
        self.times = [parse_time(r["time"]) for r in self.records]
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
        self._idf = None
        self._model = None

    @property
    def chunk_words(self):
        if self._chunk_words is None:
            self._chunk_words = [content_words(t) for t in self.chunk_text]
        return self._chunk_words

    def idf(self, word):
        """Rarity of a word across records: rare words weigh more in keyword matching."""
        if self._idf is None:
            counts = Counter()
            for r in self.records:
                counts.update(content_words(r["text"]))
            self._idf = (counts, len(self.records))
        counts, n = self._idf
        return math.log((n + 1) / (counts.get(word, 0) + 1)) + 1

    def embed(self, text):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # slow to import, so only here

            self._model = SentenceTransformer(MODEL)
        return self._model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]

    def in_window(self, since=None, until=None):
        """Per-chunk mask of chunks whose record falls in [since, until)."""
        keep = np.array([(since is None or t >= since) and (until is None or t < until) for t in self.times])
        return keep[self.chunk_record]


def day_window(corpus, day):
    """The time span of one AI Village day, from the day number in the records' links."""
    times = [t for r, t in zip(corpus.records, corpus.times) if re.search(rf"[?&]day={day}(&|$)", r["link"])]
    if not times:
        raise SystemExit(f"No records on day {day}.")
    return min(times), max(times) + timedelta(seconds=1)


def chunk_scores(claim, corpus, window):
    """Return per-chunk keyword share, fuzzy score and similarity to the expanded claim.

    Chunks outside the window get zeros.
    """
    words = content_words(claim)
    keyword = np.zeros(len(corpus.chunk_text))
    if words:
        weight = {w: corpus.idf(w) for w in words}
        total = sum(weight.values())
        need = 1 if len(words) == 1 else 2
        for c in np.flatnonzero(window):
            shared = words & corpus.chunk_words[c]
            if len(shared) >= need:
                keyword[c] = sum(weight[w] for w in shared) / total

    # Near-copies: look for the claim inside chunks at least as long as it, and
    # compare whole texts for shorter chunks, which would otherwise score 100
    # just for being a piece of the claim.
    inside = np.flatnonzero(window)
    texts = [corpus.chunk_text[c] for c in inside]
    fuzzy = np.zeros(len(corpus.chunk_text))
    if texts:
        partial = process.cdist([claim], texts, scorer=fuzz.partial_ratio, processor=default_process, workers=-1)[0]
        whole = process.cdist([claim], texts, scorer=fuzz.ratio, processor=default_process, workers=-1)[0]
        fuzzy[inside] = np.where([len(t) >= len(claim) for t in texts], partial, whole)

    query = corpus.embed(claim)
    first = np.where(window, (corpus.vectors @ query)[corpus.chunk_vector], 0.0)
    candidates = np.flatnonzero(window & ((keyword >= KEYWORD_MIN) | (fuzzy >= FUZZY_MIN) | (first >= SEED_MIN)))
    similarity = first
    if len(candidates):
        seeds = candidates[np.argsort(-first[candidates])[:SEEDS]]
        expanded = query + corpus.vectors[corpus.chunk_vector[seeds]].mean(axis=0)
        expanded /= np.linalg.norm(expanded)
        similarity = np.where(window, (corpus.vectors @ expanded)[corpus.chunk_vector], 0.0)
    return keyword, fuzzy, similarity


def find_matches(claim, corpus, since=None, until=None):
    """Return every record that matches the claim, sorted by time.

    Each match keeps its best chunk: the excerpt, its span in the record text,
    its tier, the methods that fired on it and their scores.
    """
    window = corpus.in_window(since, until)
    keyword, fuzzy, similarity = chunk_scores(claim, corpus, window)
    fired = window & ((keyword >= KEYWORD_MIN) | (fuzzy >= FUZZY_MIN) | (similarity >= EMBED_MIN))
    claim_numbers = numbers(claim)
    predicates = predicate_words(claim)

    best = {}
    for c in np.flatnonzero(fired):
        rec = int(corpus.chunk_record[c])
        methods = [name for name, hit in (("keyword", keyword[c] >= KEYWORD_MIN),
                                          ("fuzzy", fuzzy[c] >= FUZZY_MIN),
                                          ("embedding", similarity[c] >= EMBED_MIN)) if hit]
        same_numbers = claim_numbers <= numbers(corpus.chunk_text[c])
        core = (same_numbers and says_enough(predicates, corpus.chunk_words[c])
                and (similarity[c] >= CORE_MIN or fuzzy[c] >= FUZZY_MIN))
        strong = core or (same_numbers and (methods != ["embedding"] or similarity[c] >= EMBED_STRONG))
        rank = (core, strong, len(methods), float(similarity[c]))
        if rec not in best or rank > best[rec]["rank"]:
            best[rec] = {
                "rank": rank,
                "record": corpus.records[rec],
                "core": bool(core),
                "strong": bool(strong),
                "methods": methods,
                "keyword": round(float(keyword[c]), 2),
                "fuzzy": round(float(fuzzy[c])),
                "similarity": round(float(similarity[c]), 3),
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
