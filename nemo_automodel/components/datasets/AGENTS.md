# `nemo_automodel/components/datasets/` — data pipelines

Adds to the repository root `AGENTS.md`.

Per-domain pipelines under `llm/`, `vlm/`, `multimodal/`, `audio/`,
`diffusion/`, and `dllm/`, plus shared machinery: `loader.py`,
`lazy_mapped_dataset.py`, `datum.py`, `reservoir_sampler.py`, and `utils.py`.

## Keep preprocessing lazy

Dataset construction runs on every rank at job start, inside the job's wall
clock. Eager per-example work in a `map()` — image decode, tokenization of the
full corpus, media loading — turns into minutes of startup and shows up as a CI
timeout rather than as a data bug. Prefer `with_transform`/lazy access so work
happens per batch in the dataloader workers.

When you change a transform's laziness, check both startup time and that
validation metrics are unchanged; some lazy formulations alter what the
collater receives.

## Collation and packing

- The collater output is the model's input contract. Document tensor shapes and
  axis order (including which axis is batch and which is sequence) in the
  collater docstring.
- Sequence packing is not universally supported. THD packing requires model
  support; padded (BSHD) batching works everywhere. Gate on the model
  capability flag rather than on the model name.
- Packing changes the meaning of per-token loss normalization. If you touch
  packing, re-check the loss reduction with it.

## Determinism

Shuffling, sampling, and packing must be reproducible from the config seed and
must produce the same partition on every rank that needs agreement. A sampler
that differs across ranks silently changes the effective batch.
