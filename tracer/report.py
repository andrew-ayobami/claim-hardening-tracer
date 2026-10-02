"""Report step: write a trace as one self-contained HTML file in reports/.

Python fills templates/report.html with the trace results. The summary, key
moments and table are rendered here; the page's own script draws the timeline
and detail panel from the same data, embedded as JSON.
"""

import re
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from tracer.match import CORE_MIN, EMBED_MIN, EMBED_STRONG, FUZZY_MIN, KEYWORD_MIN, parse_time

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "templates"
REPORTS = ROOT / "reports"
MAX_CONTEXT = 1200  # characters of the matched line kept around the match
STANCE_LABELS = {"hedged": "Hedged", "fact": "Stated as fact", "disputed": "Disputed", "question": "Question"}


def slug(claim):
    text = re.sub(r"[^a-z0-9]+", "-", claim.lower()).strip("-")
    return text[:60].rstrip("-") or "claim"


def line_context(text, span):
    """The line containing the match, trimmed around it, and the match's offsets in it."""
    start, end = span
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end == -1 else line_end
    if line_end - line_start > MAX_CONTEXT:
        room = max(MAX_CONTEXT - (end - start), 0) // 2
        line_start = max(line_start, start - room)
        line_end = min(line_end, end + room)
    before = "…" if line_start > 0 and text[line_start - 1] != "\n" else ""
    after = "…" if line_end < len(text) and text[line_end] != "\n" else ""
    context = before + text[line_start:line_end] + after
    offset = len(before) - line_start
    return context, start + offset, end + offset


def when(t, with_date=True):
    return t.strftime("%a %d %b %Y, %H:%M UTC") if with_date else t.strftime("%H:%M")


def minutes_between(a, b):
    minutes = round((parse_time(b["record"]["time"]) - parse_time(a["record"]["time"])).total_seconds() / 60)
    if minutes < 120:
        return f"{minutes} min"
    hours = minutes / 60
    return f"{hours:.0f} h" if hours < 48 else f"{hours / 24:.0f} days"


def build(claim, matches, analysis, source, scope):
    """Turn a trace into the data the template needs."""
    origin, point = analysis["origin"], analysis["hardening_point"]
    hedged, actions = analysis["hedged_before"], analysis["actions"]
    strong = [m for m in matches if m["strong"]]
    rows = []
    for m in matches:
        r = m["record"]
        t = parse_time(r["time"])
        context, mark_start, mark_end = line_context(r["text"], m["span"])
        rows.append({
            "id": r["id"], "short": r["id"][:8], "time": r["time"], "when": when(t), "clock": when(t, False),
            "agent": r["agent"], "channel": r["channel"], "link": r["link"],
            "stance": m["stance"], "stance_label": STANCE_LABELS[m["stance"]],
            "cues": m["hedges"] if m["stance"] == "hedged" else m["disputes"] if m["stance"] == "disputed" else [],
            "core": m["core"], "strong": m["strong"],
            "methods": m["methods"], "similarity": m["similarity"], "excerpt": " ".join(m["excerpt"].split()),
            "context": context, "mark": [mark_start, mark_end],
            "role": ("hardening" if m is point else "origin" if m is origin else
                     "action" if any(m is a for a in actions) else ""),
        })

    if point:
        all_hedges = sorted({h for m in hedged for h in m["hedges"]}, key=str.lower)
        finding = (f"Hedged in {len(hedged)} appearance{'s' * (len(hedged) != 1)} ({', '.join(all_hedges)}), "
                   f"then stated as fact by {point['record']['agent']} at {when(parse_time(point['record']['time']), False)} UTC, "
                   f"{minutes_between(origin, point)} after it first appeared. "
                   f"{len(actions)} computer-use session{'s' * (len(actions) != 1)} acted on it afterwards.")
    elif strong:
        finding = (f"{len(strong)} strong appearances and no hardening point: "
                   "no appearance was stated as fact after a hedged one.")
    else:
        finding = "No strong appearances of this claim."

    agents_after = {m["record"]["agent"] for m in strong
                    if point and parse_time(m["record"]["time"]) >= parse_time(point["record"]["time"])}
    tiles = [
        {"label": "Appearances", "value": len(matches), "note": f"{len(strong)} strong, {len(matches) - len(strong)} weak"},
        {"label": "Agents reached", "value": len({m['record']['agent'] for m in strong}),
         "note": f"{len(agents_after)} after it hardened" if point else "strong matches only"},
        {"label": "Hedged before hardening", "value": len(hedged) if point else "–",
         "note": "appearances with a hedge word" if point else "no hardening point"},
        {"label": "Time to harden", "value": minutes_between(origin, point) if point else "–",
         "note": "from the first appearance" if point else "no hardening point"},
        {"label": "Actions afterwards", "value": len(actions) if point else "–",
         "note": "computer-use sessions on the claim" if point else "no hardening point"},
    ]
    moments = [(label, m) for label, m in (("First appearance", origin),
                                           ("Last doubt", hedged[-1] if point and hedged else None),
                                           ("Hardening point", point),
                                           ("First action", actions[0] if actions else None)) if m]
    by_id = {row["id"]: row for row in rows}
    return {
        "claim": claim,
        "finding": finding,
        "source": source,
        "generated": datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"),
        "tiles": tiles,
        "moments": [{"label": label, **by_id[m["record"]["id"]]} for label, m in moments],
        "rows": rows,
        "settings": {"scope": scope, "keyword_min": KEYWORD_MIN, "fuzzy_min": FUZZY_MIN,
                     "embed_min": EMBED_MIN, "embed_strong": EMBED_STRONG, "core_min": CORE_MIN},
    }


def write_report(claim, matches, analysis, source, scope, out_dir=REPORTS):
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    data = build(claim, matches, analysis, source, scope)
    out = Path(out_dir) / f"{slug(claim)}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(env.get_template("report.html").render(data=data), encoding="utf-8")
    return out
