# Automodel Researcher

An autonomous AI agent that tunes NeMo AutoModel fine-tuning configs to minimize validation loss without human intervention. You point it at a notebook and YAML config; it runs a loop—baseline, GPU saturation probe, then hyperparameter optimization—until the experiment **budget** is exhausted.

Full behavior (paths, permissions, logging schema, loop rules, final notebook) lives in [`program.md`](program.md).

## Artifacts

The agent records every run in:

| File | Role |
| --- | --- |
| `results.tsv` | Tab-separated ledger: one row per experiment with metrics, status, and short comments. |
| `journal.md` | Chronological Markdown narrative: setup, hypotheses, surprises, crashes, and why each decision was made. |
