"""Label the tracer's matches with an open-weight language model.

Runs on any machine with a GPU and transformers, such as a free Kaggle notebook
with "GPU T4 x2" (see kaggle/README.md):

    python judge.py candidates.jsonl labels.jsonl [--model Qwen/Qwen3-8B]

candidates.jsonl comes from `python trace.py export`. Copy labels.jsonl back and
pass it to `trace.py check --labels` or `trace.py trace --labels`. The answer
letters are explained in tracer/judge.py.

The model doesn't generate text: the script reads its next-token scores for the
five answer letters and takes the highest, so every candidate gets exactly one
valid answer. Labels are appended as they're made, so a stopped run resumes.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
LETTERS = "ABCDE"
# Scored as letters, the model leans to "A" whenever it's unsure ("A" is also a
# common first word), so it rarely answers B: tracer/judge.py uses only its E
# and C answers. Answering with words (fact, doubt, false, question, other) plus
# worked examples did worse on the hand-checked rows (113 of 192 right, not 125).

PROMPT = """You are checking how a message from an AI agent relates to one claim.

Claim: {claim}

Agent: {agent} ({channel})
Matched sentence: {sentence}
The sentence's line in the message: {context}

Judge the matched sentence, using the line around it for context. Which answer fits best?

A. It states the claim as true, or acts on it as already true (for example, a plan that depends on it).
B. It states the claim with doubt: as possible, likely, suspected, unconfirmed, or only someone else's report.
C. It denies or corrects the claim, or reports a check that found it false.
D. It asks whether the claim is true.
E. It doesn't express this claim: it's about a related event, a different claim on the same topic, or an outcome that says nothing about the claim.

A restatement counts as the claim even in other words or with added detail. A claim about only one part or one case of it doesn't. Doubt, denial or questions about something else in the line don't count.

Answer with one letter."""


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def letter_ids(tokenizer):
    """Token ids that can spell each answer letter at the start of the reply ("A" and " A")."""
    ids = {}
    for letter in LETTERS:
        variants = {tokenizer.encode(letter, add_special_tokens=False)[0],
                    tokenizer.encode(" " + letter, add_special_tokens=False)[-1]}
        ids[letter] = sorted(variants)
    return ids


def prompt_text(tokenizer, candidate):
    messages = [{"role": "user", "content": PROMPT.format(**candidate)}]
    # enable_thinking=False turns off Qwen3's reasoning block; other chat templates ignore it.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, help="label at most this many candidates (for a quick test)")
    args = parser.parse_args()

    candidates = load_jsonl(args.candidates)
    done = {row["id"] for row in load_jsonl(args.labels)}
    todo = [c for c in candidates if c["id"] not in done][: args.limit]
    print(f"{len(candidates)} candidates, {len(done)} already labelled, {len(todo)} to go")
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        # bf16 needs compute capability 8 (A100 and later). A T4 reports it as supported
        # but only emulates it, about 9 times slower than float16.
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    else:
        print("No GPU found: running on the CPU, which is only practical for a small model and a few candidates.")
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    ids = letter_ids(tokenizer)
    device = next(model.parameters()).device

    # Similar lengths batch together with less padding.
    todo.sort(key=lambda c: len(c["context"]) + len(c["sentence"]))
    start, labelled = time.time(), 0
    with args.labels.open("a", encoding="utf-8") as out:
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            inputs = tokenizer([prompt_text(tokenizer, c) for c in batch], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                logits = model(**inputs, logits_to_keep=1).logits[:, -1, :].float()
            for c, row in zip(batch, logits):
                scores = {letter: max(row[t].item() for t in ids[letter]) for letter in LETTERS}
                if any(math.isnan(s) for s in scores.values()):
                    raise SystemExit("The model returned NaN scores; try --model with a bf16-capable GPU or another model.")
                top = max(scores.values())
                total = sum(math.exp(s - top) for s in scores.values())
                probs = {letter: round(math.exp(s - top) / total, 3) for letter, s in scores.items()}
                answer = max(probs, key=probs.get)
                out.write(json.dumps({"id": c["id"], "answer": answer, "probs": probs, "model": args.model}) + "\n")
            out.flush()
            labelled += len(batch)
            rate = labelled / (time.time() - start)
            print(f"{labelled}/{len(todo)} labelled, {rate:.1f} per second, "
                  f"about {(len(todo) - labelled) / rate / 60:.0f} min left", flush=True)
    print(f"Done: labels in {args.labels}")


if __name__ == "__main__":
    main()
