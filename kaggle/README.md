# Running the judge on Kaggle

The language-model judge (`tracer/judge.py`) needs a GPU, which a free Kaggle notebook provides. Only the matched sentences are uploaded, never the whole dataset.

It's optional. On the seven fresh test claims it left every hardening point unchanged (see "How well it works" in the main README), but it does keep neighbouring claims out of a trace and catch denials the word list misses.

1. **Export the matches** on your own machine:

   ```bash
   python trace.py export test_cases.json fresh_cases.json fresh_round2.json
   ```

   This writes `data/judge/candidates.jsonl`. For a single claim, use `python trace.py export --claim "..." --day 314`.

2. **Make a private Kaggle dataset.** Sign in at kaggle.com and verify your phone number under Settings (notebooks need it for GPUs and internet). Then go to Create > New Dataset, upload `candidates.jsonl` and `kaggle/judge.py`, and keep it private. The AI Village data is gated, so don't make it public.

3. **Make a notebook.** Go to Create > New Notebook. In the right-hand panel, set Accelerator to **GPU T4 x2** and turn **Internet** on. Under Input, add your dataset.

4. **Run the judge** in a code cell:

   ```python
   import glob
   d = glob.glob("/kaggle/input/**/judge.py", recursive=True)[0].rsplit("/", 1)[0]
   !python {d}/judge.py {d}/candidates.jsonl /kaggle/working/labels.jsonl --limit 40
   ```

   That labels 40 matches as a test. The first run downloads the model (Qwen3-8B, about 16 GB, a few minutes on Kaggle). Run it again without `--limit` to label the rest. It carries on where it stopped and prints the time left: about 25 minutes for 5,600 matches on two T4s. Keep the tab open while it runs.

5. **Bring the labels back.** Run this in a new cell and click the link it shows (the Output panel's file list doesn't always refresh):

   ```python
   from IPython.display import FileLink
   FileLink("labels.jsonl")
   ```

   Put the downloaded `labels.jsonl` in `data/judge/`. Then:

   ```bash
   python trace.py check test_cases.json fresh_cases.json fresh_round2.json --labels data/judge/labels.jsonl
   python trace.py trace "social media is network-blocked for all agents" --day 314 --labels data/judge/labels.jsonl
   ```

To try another open-weight model, add `--model <hugging face id>` to the `judge.py` command.
