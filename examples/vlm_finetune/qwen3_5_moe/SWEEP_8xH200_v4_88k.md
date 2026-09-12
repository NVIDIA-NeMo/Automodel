# Config sweep — `qwen3_6_35b_v4_88k_ep8.yaml` on 8 × H200

Companion to `RUNBOOK_v4_88k.md` and `BRINGUP_2xB300_v4_88k.md`. The runbook says what
to run; the bring-up log records getting the recipe working at all. This file records
the batch-size / activation-checkpointing / batching-order sweep that set the values now
in the YAML, and the measurements that rejected the alternatives. Written 2026-09-12.

**Hardware:** one node, 8 × NVIDIA H200 SXM (140.4 GiB usable each, 143,771 MiB as
`nvidia-smi` reports it). FSDP2 + `ep_size: 8` + DeepEP, `tp/cp/pp = 1`, so `dp_size = 8`
and `grad_accum = global_batch_size / (local_batch_size * 8)`.

**Corpus:** `data/v4_88k_filtered/train.parquet`, 87,279 rows after prefiltering, mean
~9.7k tokens, max 40,921. At `global_batch_size: 32` that is ~2,727 optimizer steps per
epoch regardless of `local_batch_size`.

---

## 1. How to read throughput numbers here

The recipe logs two token counts on every step and they measure different things
(`recipes/vlm/finetune.py`):

```python
# number of tokens in the batch, excluding any tail padding.
num_tokens_in_batch = sum(batch["labels"].numel() - count_tail_padding(batch["labels"]) ...)
tps = num_tokens_in_batch / time_delta
```

- `tps` — **sequence positions per second summed over all 8 ranks**, interior padding
  included. This is the big number (~20-50k).
- `num_label_tokens` — supervised tokens only (`labels != -100`), ~10k per step. This is
  what the loss is averaged over. It is *not* the numerator of `tps`.

Two consequences, both of which produced misleading readings during this sweep:

1. **`tps` is only comparable between runs on the same data.** Longer sequences amortize
   the per-step fixed costs (optimizer step over 35B params, FSDP all-gather /
   reduce-scatter, EP all-to-all latency, kernel launches), so the same config reports
   ~48k tok/s on a corpus of 40k-token rows and ~22k tok/s on the real corpus. Never
   compare a `longest256` figure against a real-corpus figure.
2. **`tps` understates length-grouped batching**, because it counts padding as work and
   grouping's entire benefit is removing padding.

**Use `s/step` as the cross-config metric.** Every step consumes exactly
`global_batch_size = 32` samples no matter how they are batched or padded, so s/step is
directly proportional to wall-clock time per epoch. `label tok/s` is the best proxy for
useful learning per second.

---

## 2. Results — real corpus, 50 steps each

Identical otherwise: EP8, gbs 32, warmup 50 / decay 5456, val every 25. s/step excludes
warmup steps 0-1 and any step that absorbed a validation or checkpoint save.

| `local_batch_size` | batching | AC | s/step | samp/s | positions/s | label tok/s | torch peak | smi peak | val @24 | val @49 | grad_norm max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | length-grouped | full | **8.01** | 3.99 | 43,114 | 1,242 | 106.4 GiB | 131.3 GiB | 0.6701 | 0.5759 | 11.72 |
| 2 | length-grouped | full | 9.79 | 3.27 | 38,261 | 1,073 | 75.3 GiB | 88.8 GiB | 0.6642 | 0.5718 | 13.28 |
| **2** | **random** | **full** | **16.62** | 1.92 | 22,241 | 604 | 72.4 GiB | 86.9 GiB | 0.6561 | **0.5605** | 8.50 |
| 4 | random | full | 18.97 | 1.69 | 19,493 | 529 | 105.4 GiB | 122.2 GiB | 0.6556 | 0.5603 | 8.27 |
| 2 | random | selective | 16.2 † | — | — | — | 112.5 GiB | 130.4 GiB | 0.6560 | — | — |

† partial run, stopped at step 25 once the trade was clear.

The bold row is what the YAML ships. Loss curves are indistinguishable across all arms
at matched steps (step 0 is 0.938 in every random-batching arm), and no run showed loss
spikes, NaNs, or router collapse. Gradient clipping at `max_norm: 1.0` fires on **every**
step in every arm, so the clip — not the schedule — sets the effective step size; a lower
peak LR than `1.0e-5` is worth testing.

### Why `local_batch_size: 2` over 1 and 4

- **vs 1:** 16.62 vs 19.53 s/step, same validation loss (0.5605 vs 0.5608). Free.
- **vs 4 (random):** lbs 4 is *slower* — 18.97 s/step — and uses 33 GiB more. The cause
  is padding: the collator pads to the longest sample in each micro-batch, so padding to
  the longest of four wastes more than the larger batch gains. This only reverses when
  length grouping removes the padding (see §4).

### Why full activation checkpointing over `selective`

`ActivationCheckpointingMode` is `bool | "full" | "selective"` — there is no intermediate
rung. For this model `selective` bought **2.4%** (16.2 vs 16.62 s/step) for **+40 GiB**,
and on worst-case batches it reaches 139.6 GiB of 140.4. Rejected. `activation_checkpointing_scope`
does not help here: the vision tower is frozen and already skipped.

---

## 3. Worst-case memory — `longest256.parquet`

`data/v4_88k_filtered/longest256.parquet` holds the 256 longest rows (38,767-40,921
tokens each), so *every* batch is a worst case. 8 steps × 32 samples covers all 256 rows
exactly once. **Fit test only — do not quote throughput from this dataset** (§1).

| config | padded tokens/rank | torch peak | smi peak | result |
|---|---|---|---|---|
| lbs 2, random, full AC | 81,842 | 76.8 GiB | 96.9 GiB (69%) | fits, comfortable |
| lbs 4, random, full AC | ~158,792 | 114.5 GiB | 138.7 GiB (98.8%) | fits, no margin |
| lbs 4, length-grouped, full AC | 163,684 | 115.1 GiB | 139.5 GiB (**99.4%**) | fits, no margin |
| lbs 2, selective AC | 81,842 | 121.4 GiB | 139.6 GiB (99.6%) | fits, no margin |
| lbs 4, selective AC | ~158,792 | — | — | **OOM** |

`nvidia-smi` runs ~20-24 GiB above `torch.cuda.max_memory_allocated` — NCCL and DeepEP
buffers, cuDNN/cuBLAS workspaces, and the CUDA context. None of it is reclaimable under
pressure, so **judge headroom by the smi figure, not torch's.**

A behavioural difference worth knowing: with random batching the smi footprint *climbs*
toward its ceiling over successive steps (136.6 → 138.7 and still rising at step 8);
with length grouping it reaches the ceiling at step 1 and stays pinned there, because
the sampler places the longest rows together by construction rather than by luck.

---

## 4. Length-grouped batching

`dataloader.length_grouped_sampler` (see the commented block in the YAML) sorts samples
into length-homogeneous batches, eliminating padding. It is the single largest speedup
found: **1.70× at lbs 2, 2.08× at lbs 4** — one epoch drops from ~12.6 h to ~6.1 h.

It is shipped **disabled**, for two reasons:

1. **Validation loss was consistently worse** — 0.5718 (lbs 2) and 0.5759 (lbs 4) against
   0.5605 for random batching, at both val points in both arms. A length-homogeneous
   batch averages its gradient over a narrower slice of the corpus. The evidence is
   weak, though: one seed, 50 steps (1.8% of an epoch), and the entire comparison ran
   inside LR warmup so neither arm reached the production schedule. **A ~300-step A/B
   would settle whether this gap is real; it has not been run.**
2. **Grouping makes the worst-case batch certain rather than rare.** Measured at lbs 4
   (§3); **unmeasured at lbs 2**, where the guaranteed worst batch is 81,842 padded
   tokens/rank.

`lbs 4 + length grouping` is the fastest configuration measured and it does survive the
worst batch the corpus can produce — but at 99.4% of the card, with a checkpoint save at
that occupancy never tested. A state-dict gather allocates while the training peak is
resident; on real data with checkpointing live the same config sat at 131.3 GiB, and
those two peaks have not been observed together. Do not adopt it without that test.

---

## 5. Verified operational facts

- **Checkpointing and validation both work at lbs 4.** A 50-step run with
  `ckpt_every_steps: 25` and `val_every_steps: 25` completed clean: sharded save ~80 GB
  in ~64 s, final consolidated HF export 26 shards / ~67 GiB. Validation adds no memory
  beyond the training peak (it runs under `no_grad` on normal-length rows).
- **`save_consolidated: final` logs `v4_compatible=False`.** The export may not load
  under transformers v4 without `--checkpoint.v4_compatible=True`.
- **Killing a run needs SIGKILL.** Ranks re-parent to PID 1 and hold ~100 GB of GPU
  memory through SIGTERM; the next launch then dies on `EADDRINUSE` (port 29500) while
  the memory sampler silently records the *previous* run's peak as if it were the new
  one's. Always drain to <2000 MiB and confirm port 29500 is free before measuring.

---

## 6. Reproducing

```bash
# Worst-case fit test (8 steps = all 256 longest rows, no checkpointing)
automodel examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_v4_88k_ep8.yaml \
  --nproc-per-node 8 --distributed.ep_size 8 \
  --step_scheduler.local_batch_size 4 --step_scheduler.global_batch_size 32 \
  --step_scheduler.max_steps 8 --lr_scheduler.lr_warmup_steps 1 \
  --checkpoint.enabled false \
  --dataset.path_or_dataset data/v4_88k_filtered/longest256.parquet

# 50-step production-shape run (checkpointing and validation live)
automodel examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_v4_88k_ep8.yaml \
  --nproc-per-node 8 --distributed.ep_size 8 \
  --step_scheduler.max_steps 50 \
  --step_scheduler.val_every_steps 25 --step_scheduler.ckpt_every_steps 25 \
  --lr_scheduler.lr_warmup_steps 50 --lr_scheduler.lr_decay_steps 5456
```

Sample `nvidia-smi` alongside the run; `torch.cuda.max_memory_allocated` (the `mem`
field in the log line) is not the whole footprint.

---

## 7. Open questions

1. **~300-step A/B of length-grouped vs random at lbs 2** — the only test that decides
   whether the 1.70× speedup is affordable. Everything else about grouping is measured.
2. **Worst-case fit for lbs 2 + length grouping** (~5 min) — required before enabling it.
3. **Checkpoint save at 139.5 GiB occupancy for lbs 4 + grouping** (~10 min) — required
   before that config could be considered.
4. **Peak LR below `1.0e-5`.** Clipping fires on every step of every arm measured.
