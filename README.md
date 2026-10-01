# Claim hardening tracer

When a swarm acts on something false, investigators need to know where it came from and where the doubt got lost. This tool traces a claim through chat, memory and actions, and flags the moment it hardened into fact.

Work in progress for the AI swarms hackathon.

## How it works

1. **Load:** turn the dataset tables into one list of records (time, agent, channel, text, link).
2. **Index:** prepare the records for search once and save the result to disk.
3. **Trace:** find every appearance of a claim, sort them by time, and mark each as hedged or stated as fact.
4. **Report:** write one HTML file with the timeline.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

## Layout

| Folder | Contents |
| --- | --- |
| `loaders/` | Code that turns a dataset's tables into records |
| `tracer/` | Index, trace and hedge detection |
| `templates/` | HTML report template |
| `demo_data/` | Fake demo dataset, safe to commit |
| `reports/` | Generated HTML reports |
| `data/` | Real dataset files. Ignored by git and never committed |

## Data

Built on the AI Village dataset from AI Digest. Real data stays in `data/`, which git ignores.
