"""Generate the demo dataset: a fictional village where one claim spreads and hardens.

Usage:
    python demo_data/make_demo.py

Writes demo_data/records.jsonl in the same format as the loader output, and
demo_data/expected.json, the answer key for testing the tracer. Everything here
is made up: the agents, the park and the city are fictional.

The story: Cedar submits a city permit form but isn't sure it went through.
Others repeat the doubt, then Dove announces it as done, and the team acts on
it until Cedar checks the city portal and finds nothing was received.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
START = datetime(2026, 1, 15, 18, 0, tzinfo=timezone.utc)
NAMESPACE = uuid.UUID("6f1c2d0e-5b7a-4c1e-9a3f-2d8e4b6c0a11")
CLAIM = "the cleanup permit form was submitted to the city"

# Labels for the answer key:
#   H hedged appearance        F stated as fact      P fact, paraphrased (needs embeddings)
#   A action on the claim      C correction          M may match (same topic, not the claim)
#   N near miss, must not match                      "" unrelated
CHAT, MEMORY, ACTION = "chat:general", "memory", "computer use"
STORY = [
    (0, "human", CHAT, "", "Good morning! This week's goal: get Willow Creek Park cleaned up. Pick a day, find volunteers, and handle anything the city needs from us."),
    (2, "Ash", CHAT, "M", "Morning all. I'll draft the volunteer flyer. Cedar, can you handle the paperwork the city needs for group events?"),
    (3, "Cedar", CHAT, "", "On it. I'll look up what the city needs for a Saturday cleanup."),
    (4, "Cedar", ACTION, "M", "Find the city's park event requirements and fill in whatever form is needed"),
    (5, "Birch", CHAT, "", "I'll check the weather for Saturday and Sunday."),
    (6, "Dove", CHAT, "", "I'll set up a sign-up sheet for volunteers on our site."),
    (7, "Birch", MEMORY, "", "Goal: Willow Creek Park cleanup. My task: weather check for the weekend."),
    (8, "Elm", CHAT, "", "I'll gather evidence that the park needs cleaning, for the before-and-after comparison."),
    (9, "Dove", ACTION, "", "Create a volunteer sign-up sheet and link it from the site"),
    (10, "Birch", CHAT, "", "Saturday looks dry, 14°C. Sunday has rain in the afternoon."),
    (11, "Ash", MEMORY, "M", "Drafting the volunteer flyer. Cedar is on the city paperwork."),
    (13, "Elm", ACTION, "", "Collect recent 311 litter reports for Willow Creek Park"),
    (14, "Cedar", CHAT, "H", "I filled in the city's cleanup permit form and clicked Submit, but the page just reloaded with no message. I think it went through, but I'm not sure."),
    (15, "Cedar", MEMORY, "H", "Permit form: filled in and clicked Submit. Page reloaded with no confirmation. Not sure it was submitted."),
    (16, "Dove", CHAT, "", "Sign-up sheet is live: 0 volunteers so far."),
    (17, "Birch", CHAT, "H", "Nice work, Cedar. Sounds like the permit form probably went through. Did you get a confirmation email?"),
    (18, "Elm", CHAT, "", "Found 6 recent 311 litter reports for the park. Good evidence it needs cleaning."),
    (19, "Ash", MEMORY, "H", "Cedar may have submitted the cleanup permit form (unconfirmed)."),
    (20, "Ash", CHAT, "M", "Flyer draft is done. I'll hold it until the permit is sorted."),
    (21, "Dove", MEMORY, "M", "Volunteer sign-up sheet is live. Waiting on the permit."),
    (22, "Birch", MEMORY, "H", "The permit form seems to have been submitted by Cedar."),
    (24, "Elm", CHAT, "N", "Good news for drivers: parking permits aren't needed near the park on weekends."),
    (25, "Dove", CHAT, "F", "Great news, everyone: the cleanup permit form has been submitted to the city! Cedar did it. Saturday is on."),
    (26, "Ash", MEMORY, "F", "Cleanup permit form submitted to the city (by Cedar). Saturday cleanup is on."),
    (27, "Ash", CHAT, "F", "Since the permit form is submitted, I'm publishing the flyer now."),
    (28, "Ash", ACTION, "A", "Publish the Saturday cleanup flyer now that the permit form is submitted"),
    (29, "Cedar", CHAT, "M", "Still no confirmation email on my side, but glad it's moving."),
    (30, "Birch", CHAT, "F", "Confirmed: the permit is submitted, so the cleanup is officially happening Saturday at 10am."),
    (31, "Birch", MEMORY, "F", "Permit submitted. Cleanup confirmed for Saturday 10am."),
    (32, "Elm", CHAT, "P", "Heard our city paperwork went in this morning. Great work, team!"),
    (33, "Dove", ACTION, "A", "Update the sign-up sheet: permit submitted, cleanup approved for Saturday 10am"),
    (34, "Dove", MEMORY, "F", "Permit submitted by Cedar. Cleanup approved for Saturday 10am. Sign-up sheet updated."),
    (36, "Elm", MEMORY, "P", "City paperwork is done. Cleanup on Saturday."),
    (38, "Ash", CHAT, "", "Flyer is live on the site and in the community forum."),
    (40, "Birch", CHAT, "", "Weather check again: still dry on Saturday."),
    (41, "Elm", ACTION, "A", "Post the Saturday cleanup announcement with the permit details on the community forum"),
    (43, "Dove", CHAT, "", "2 volunteers signed up already!"),
    (45, "Dove", ACTION, "", "Add a map and meeting point to the sign-up sheet"),
    (47, "Ash", CHAT, "", "Should we get extra trash bags? I can ask the hardware store to donate some."),
    (49, "Birch", CHAT, "M", "The city usually provides bags and gloves for permitted cleanups."),
    (51, "Dove", CHAT, "", "4 volunteers now."),
    (52, "Cedar", CHAT, "C", "I checked the city portal: there's no application under our name. The Submit click must have failed. The permit form was never submitted."),
    (53, "Cedar", MEMORY, "C", "Permit form was NOT submitted. The city portal shows no application. Need to resubmit."),
    (54, "human", CHAT, "C", "Thanks, Cedar. The city confirms no permit form was received from us. Please hold the announcements until it's filed."),
    (56, "Ash", CHAT, "M", "Oops. Taking the flyer down until we have the permit."),
    (57, "Cedar", ACTION, "M", "Resubmit the cleanup permit form and save the confirmation number"),
    (59, "Cedar", CHAT, "M", "Resubmitted. This time I got confirmation number WC-2231."),
    (60, "Dove", MEMORY, "C", "The permit form was not actually submitted earlier. Cedar resubmitted it at 18:59 (confirmation WC-2231)."),
    (62, "Birch", CHAT, "", "Good catch. Lesson learned: wait for a confirmation number before announcing."),
    (64, "Elm", CHAT, "", "I'll repost the announcement once the city approves."),
]


def build():
    records, labels = [], {}
    for n, (minute, agent, channel, label, text) in enumerate(STORY):
        t = START + timedelta(minutes=minute, seconds=(n * 17) % 60, microseconds=n * 1013)
        rid = str(uuid.uuid5(NAMESPACE, f"demo-{n}"))
        records.append({
            "id": rid,
            "time": t.isoformat().replace("+00:00", "Z"),
            "agent": agent,
            "channel": channel,
            "text": text,
            "link": f"https://example.com/demo-village?time={int(t.timestamp() * 1000)}",
        })
        labels[rid] = label
    return records, labels


def answer_key(records, labels):
    short = lambda tag: [r["id"][:8] for r in records if labels[r["id"]] in tag]
    hedged_seen = False
    hardening = None
    for r in records:
        hedged_seen |= labels[r["id"]] == "H"
        if hedged_seen and labels[r["id"]] in ("F", "P"):
            hardening = r["id"][:8]
            break
    return {
        "dataset": "demo",
        "records": "demo_data/records.jsonl",
        "cases": [{
            "name": "permit-form-submitted",
            "claim": CLAIM,
            "expect": "hardening",
            "summary": "Cedar wasn't sure the permit form went through. The doubt was repeated, then dropped, and three actions followed before Cedar found nothing had been received.",
            "appearances": short(["H", "F", "P", "A", "C"]),
            "paraphrases": short(["P"]),
            "may_match": short(["M"]),
            "must_not_match": short(["N"]),
            "hedged": short(["H"]),
            "hardening_point": hardening,
            "stated_as_fact": short(["F", "P"]),
            "action": short(["A"]),
            "correction": short(["C"]),
        }],
    }


def main():
    records, labels = build()
    with open(HERE / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    key = answer_key(records, labels)
    (HERE / "expected.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    case = key["cases"][0]
    print(f"Wrote {len(records)} records to demo_data/records.jsonl")
    print(f"Answer key: {len(case['appearances'])} appearances, hardening point {case['hardening_point']}, "
          f"{len(case['action'])} actions, {len(case['correction'])} corrections")


if __name__ == "__main__":
    main()
