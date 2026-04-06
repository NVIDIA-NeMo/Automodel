# PP Schedule Regression — 22% → 16.8% MFU

## Baselines

| | Container | PyTorch | MFU |
|---|---|---|---|
| **Old (target)** | `nemo-automodel:26.02.rc2` | ~2.6 | **~22%** |
| **New (regressed)** | `nemo-automodel:nightly_202604` | 2.11.0a0 | **~16.8%** |

Config: PP=4, TP=2, interleaved 1F1B, compile=True, 8×H100, Llama-3.3-70B PEFT (LoRA r=16).

---

## Root Cause

PT 2.11 changed `get_schedule_class("interleaved1f1b")` to return `_PipelineScheduleRuntime` instead of the old pure-Python `ScheduleInterleavedWith1F1B`.

The new runtime issues **152 P2P send/recv ops per ga_step** vs **80 in the old schedule** — measured directly from nsys kernel summaries.

```
n_microbatches=4, virtual_stages_per_rank=10, PP=4
Theoretical minimum: 4 × 10 × 2 = 80 (send+recv per microbatch per stage)
New runtime actual:  152  (+72 extra from warmup/cooldown handshakes)
```

The extra 72 ops are fence-style handshakes in the `_PipelineScheduleRuntime` state machine across the warmup→steady→cooldown phase transitions. These fall outside compute-overlap windows, extending pipeline bubbles on every rank.

At ga_steps=8: `(152 - 80) × 8 = 576 extra NCCL ops per training step`.

---

## What Was Ruled Out

| Hypothesis | Test | Result |
|---|---|---|
| `batch_p2p` (PyTorch PRs #175572, #178815) | Backported + benchmarked | No improvement on NVLink |
| `direct_copy_kernel_cuda` from BMM transpose | Einsum replacement benchmark | Neutral (BMM=16.84%, einsum=16.81%) |
| GA without PP | fsdp_tp run | ~26% MFU — not regressed |
| TP, FSDP, compile | All tested in isolation | Not the cause |

---

## Fix to Test

**Location:** `nemo_automodel/components/distributed/pipelining/functional.py`, line 542

```python
# Current (PT 2.11 resolves to _PipelineScheduleRuntime):
schedule_class = get_schedule_class(pipeline_parallel_schedule)

# Proposed fix: pin the old Python class for interleaved1f1b
from torch.distributed.pipelining.schedules import ScheduleInterleavedWith1F1B
if pipeline_parallel_schedule == "interleaved1f1b":
    schedule_class = ScheduleInterleavedWith1F1B
else:
    schedule_class = get_schedule_class(pipeline_parallel_schedule)
```

This forces the old 80-op schedule while leaving all other schedule types unaffected.

---

## Reproducer

```bash
# Regressed (new container):
sbatch benchmark_bmm_pp4tp2.sbatch          # → ~16.8% MFU

# Expected fix:
# Apply the patch above, then:
sbatch benchmark_bmm_pp4tp2.sbatch          # → ~22% MFU (hypothesis)

# Reference (old container):
sbatch benchmark_old_container.sbatch       # → ~22% MFU
```
