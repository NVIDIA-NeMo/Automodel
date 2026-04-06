# PP Schedule Regression — 22% → 16.8% MFU

## Baselines

| | Container | PyTorch | MFU |
|---|---|---|---|
| **Old (target)** | `nemo-automodel:26.02.rc2` | 2.9 | **~22%** |
| **New (regressed)** | `nemo-automodel:nightly_202604` | 2.11.0a0 | **~16.8%** |

Config: PP=4, TP=2, interleaved 1F1B, compile=True, 8×H100, Llama-3.3-70B PEFT (LoRA r=16).

---

## Root Cause

`get_schedule_class("interleaved1f1b")` returns `ScheduleInterleaved1F1B` in both containers — the class name is unchanged. What changed is its **implementation**:

| | 26.02.rc2 | nightly_202604 (PT 2.11) |
|---|---|---|
| `ScheduleInterleaved1F1B` base class | pure-Python standalone | **subclass of `_PipelineScheduleRuntime`** |
| P2P send/recv ops per ga_step (nsys) | **80** | **152** |

PT 2.11 silently re-implemented `ScheduleInterleaved1F1B` on top of `_PipelineScheduleRuntime`:

```python
>>> from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B
>>> [c.__name__ for c in ScheduleInterleaved1F1B.__mro__]
['ScheduleInterleaved1F1B', '_PipelineScheduleRuntime', 'PipelineScheduleMulti', ...]
```

The `_PipelineScheduleRuntime` backend uses a `pipeline_order_with_comms` state machine that adds extra send/recv handshakes across warmup→steady→cooldown phase transitions. The old pure-Python schedule overlapped these with compute; the new runtime issues them as separate serialized ops.

```
Setup: n_microbatches=4, virtual_stages_per_rank=10, PP=4
Theoretical minimum comms: 4 × 10 × 2 = 80  (send+recv per microbatch per stage boundary)
New runtime actual:         152               (+72 extra handshakes)
Extra per training step:   (152 - 80) × 8 ga_steps = 576 extra NCCL ops
```

---

## What Was Ruled Out

| Hypothesis | Test | Result |
|---|---|---|
| `batch_p2p` (PyTorch PRs #175572, #178815) | Backported + benchmarked | No improvement on NVLink |
| `direct_copy_kernel_cuda` from BMM `.t().expand()` | Einsum replacement on both fsdp_tp and PP+TP configs | Neutral — BMM=16.84%, einsum=16.81% |
| GA without PP | fsdp_tp benchmark (TP=2, FSDP2, compile) | ~26% MFU — not regressed |
| TP / FSDP2 / compile individually | Tested in isolation | Not the cause |

---

## Fix Options

`ScheduleInterleaved1F1B` in PT 2.11 inherits from `_PipelineScheduleRuntime` — there is no pure-Python fallback in the same container.

**Option A — File upstream PyTorch issue** *(recommended)*
Report to PyTorch distributed team with nsys evidence: `_PipelineScheduleRuntime`-based interleaved 1F1B regresses P2P op count from 80 → 152 vs the prior pure-Python implementation.

**Option B — Vendor the old pure-Python schedule**
Extract `ScheduleInterleaved1F1B` from `26.02.rc2`'s `schedules.py` and add it as a local class. Force it in `functional.py:542`:

```python
# nemo_automodel/components/distributed/pipelining/functional.py, line 542
from nemo_automodel.components.distributed.pipelining.legacy_schedule import LegacyScheduleInterleaved1F1B
if pipeline_parallel_schedule == "interleaved1f1b":
    schedule_class = LegacyScheduleInterleaved1F1B
else:
    schedule_class = get_schedule_class(pipeline_parallel_schedule)
```

**Option C — Stay on 26.02.rc2**
Block the container upgrade until upstream resolves the regression.

---

## Reproducer

```bash
# Current regressed state:
sbatch benchmark_bmm_pp4tp2.sbatch        # → ~16.8% MFU

# Reference (old container, confirms target):
sbatch benchmark_old_container.sbatch     # → ~22% MFU

# After Option B fix:
sbatch benchmark_bmm_pp4tp2.sbatch        # → ~22% MFU (expected)
```
