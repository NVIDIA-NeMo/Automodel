# PP Schedule Regression — 22% → 16.8% MFU

## Baselines

| | Container | PyTorch | MFU |
|---|---|---|---|
| **Old (target)** | `nemo-automodel:26.02.rc2` | 2.10.0a0+b558c986e8.nv25.11 | **~22%** |
| **New (regressed)** | `nemo-automodel:nightly_202604` | 2.11.0a0+eb65b36914.nv26.02 | **~16.8%** |

Note: both are NVIDIA nightly builds, not vanilla upstream PyTorch. Versions confirmed from `torch.__version__` inside each container.

Config: PP=4, TP=2, interleaved 1F1B, compile=True, 8×H100, Llama-3.3-70B PEFT (LoRA r=16).

---

## Root Cause

`get_schedule_class("interleaved1f1b")` returns `ScheduleInterleaved1F1B` in both containers — the class name is unchanged. What changed is its **implementation**:

| | 26.02.rc2 (PT 2.10 NVIDIA) | nightly_202604 (PT 2.11 NVIDIA) |
|---|---|---|
| `ScheduleInterleaved1F1B` MRO | `→ PipelineScheduleMulti → _PipelineSchedule` | `→ _PipelineScheduleRuntime → PipelineScheduleMulti` |
| Comm dispatch | inline `stage.get_fwd_send_ops()` + `batch_isend_irecv` per timestep | explicit `SEND_F/RECV_F/SEND_B/RECV_B` actions in `pipeline_order_with_comms` |
| P2P ops per `schedule.step()` call (nsys) | **80** | **152** |

PT 2.11 silently inserted `_PipelineScheduleRuntime` into the MRO:

```
# 26.02.rc2
ScheduleInterleaved1F1B → PipelineScheduleMulti → _PipelineSchedule → ABC

# nightly_202604
ScheduleInterleaved1F1B → _PipelineScheduleRuntime → PipelineScheduleMulti → _PipelineSchedule → ABC
```

The `_PipelineScheduleRuntime` lowering step populates `pipeline_order_with_comms` with explicit comm actions. It generates 152 ops vs the theoretical minimum of 80. The extra 72 are repeated on **every** `schedule.step()` call (i.e., every ga_step) — not a one-time overhead.

```
Setup: n_microbatches=4, virtual_stages_per_rank=10, PP=4
Theoretical minimum: 4 × 10 × 2 = 80 ops  (send+recv per microbatch per stage boundary)
New runtime actual:  152 ops               (+72 extra per schedule.step() call)
Per training step:   (152 - 80) × 8 ga_steps = 576 extra NCCL ops
```

---

## What Was Ruled Out

| Hypothesis | Test | Result |
|---|---|---|
| `batch_p2p` (PyTorch PRs #175572, #178815) | Backported + benchmarked | No improvement on NVLink |
| `direct_copy_kernel_cuda` from BMM `.t().expand()` | Einsum replacement on fsdp_tp and PP+TP configs | Neutral — BMM=16.84%, einsum=16.81% |
| GA without PP | fsdp_tp benchmark (TP=2, FSDP2, compile) | ~26% MFU — not regressed |
| TP / FSDP2 / compile individually | Tested in isolation | Not the cause |

---

## Fix Options

`ScheduleInterleaved1F1B` in PT 2.11 inherits from `_PipelineScheduleRuntime` — there is no pure-Python fallback in the same container.

**Option A — File upstream PyTorch issue** *(recommended)*
Report to PyTorch distributed team with nsys evidence: the `_PipelineScheduleRuntime`-based interleaved 1F1B regresses P2P op count from 80 → 152 vs the PT 2.10 NVIDIA implementation. Provide the MRO diff and nsys numbers as proof.

**Option B — Vendor the PT 2.10 schedule**
Copy `ScheduleInterleaved1F1B` from `26.02.rc2`'s `schedules.py` into automodel as a local class and force it in `functional.py:542`:

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
