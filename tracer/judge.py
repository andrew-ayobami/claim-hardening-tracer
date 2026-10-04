"""Language-model judge: a second opinion on the rules' calls.

The rules decide whether a sentence restates the claim (similarity, numbers and
predicate words) and how (hedge and dispute word lists). On new claims they're
fooled by neighbouring claims that read almost the same, and by doubt or denial
in words that aren't on the lists. A language model reads each match's sentence
in its line and answers one question about it:

    A  states the claim as true, or acts on it as true
    B  states the claim with doubt
    C  denies or corrects the claim
    D  asks whether the claim is true
    E  doesn't state the claim: a related event, a different claim, or an
       outcome that says nothing about it

The judge runs wherever there's a GPU: `trace.py export` writes the matches to
judge, kaggle/judge.py labels them (free on Kaggle), and `--labels` reads the
answers back. Matches without a label keep the rules' calls.

Only two answers are used. E demotes a match to weak, so a neighbouring claim
can't be the origin or hardening point, and C marks it disputed. The rest is
left to the rules: with one-token answers, Qwen3-8B said A for most matches it
wasn't sure about and B for only 32 of 5,678 matches, so it can't replace the
hedge list, and its A was too loose to make a match a close restatement.
"""

import hashlib
import json
from pathlib import Path

from tracer.harden import context

STANCES = {"A": "fact", "B": "hedged", "C": "disputed", "D": "question", "E": "other"}
MAX_CONTEXT = 700  # characters of the matched line shown to the judge


def candidate_id(claim, match):
    """A stable id for one claim and one matched span, so labels survive re-running the trace."""
    r = match["record"]
    key = f"{claim}|{r['id']}|{match['span'][0]}|{match['span'][1]}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def trimmed_line(match):
    """The matched line, cut to MAX_CONTEXT characters around the matched sentence."""
    text, (start, end) = match["record"]["text"], match["span"]
    line = context(text, match["span"], "line")
    if len(line) <= MAX_CONTEXT:
        return line
    offset = text.rfind("\n", 0, start) + 1
    middle = (start + end) // 2 - offset
    left = max(0, min(middle - MAX_CONTEXT // 2, len(line) - MAX_CONTEXT))
    return ("…" if left else "") + line[left:left + MAX_CONTEXT] + ("…" if left + MAX_CONTEXT < len(line) else "")


def candidate(claim, match):
    r = match["record"]
    return {
        "id": candidate_id(claim, match),
        "claim": claim,
        "record": r["id"][:8],
        "time": r["time"],
        "agent": r["agent"],
        "channel": r["channel"],
        "sentence": match["excerpt"],
        "context": trimmed_line(match),
    }


def export(traces, path):
    """Write one line per match to judge. traces is a list of (claim, matches). Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen, count = set(), 0
    with path.open("w", encoding="utf-8") as f:
        for claim, matches in traces:
            for m in matches:
                c = candidate(claim, m)
                if c["id"] not in seen:
                    seen.add(c["id"])
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
                    count += 1
    return count


def load_labels(path):
    """Read the judge's answers: a map from candidate id to A-E."""
    labels = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("answer") in STANCES:
                labels[row["id"]] = row["answer"]
    return labels


def apply_labels(claim, matches, labels):
    """Replace the rules' tier with the judge's answer wherever there is one. Returns how many were labelled."""
    labelled = 0
    for m in matches:
        answer = labels.get(candidate_id(claim, m))
        if answer is None:
            continue
        labelled += 1
        m["judge"] = STANCES[answer]
        if answer == "E":
            m["core"] = m["strong"] = False
    return labelled
