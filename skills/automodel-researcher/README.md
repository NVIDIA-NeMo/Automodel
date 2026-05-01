# Automodel Researcher

An autonomous AI agent that tunes NeMo AutoModel fine-tuning configs to minimize validation loss without human intervention. Set `Notebook` to your training notebook or launcher script, set `Config` to the YAML config you want to tune, and the research agent will run autonomously through the baseline, GPU saturation probe, and hyperparameter optimization loop until the experiment **budget** is exhausted.

Full behavior (paths, permissions, logging schema, loop rules, final notebook) lives in [`program.md`](program.md).

## Artifacts

The agent records every run in:

| File | Role |
| --- | --- |
| `results.tsv` | Tab-separated ledger: one row per experiment with metrics, status, and short comments. |
| `journal.md` | Chronological Markdown narrative: setup, hypotheses, surprises, crashes, and why each decision was made. |
