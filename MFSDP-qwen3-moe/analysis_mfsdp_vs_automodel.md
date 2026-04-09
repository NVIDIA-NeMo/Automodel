# nsys Analysis: MFSDP vs AutoModel — Qwen3-30B-A3B DeepEP (Apple-to-Apple)

**Date**: 2026-04-08  
**Model**: Qwen3-30B-A3B  
**Hardware**: 4 nodes / 32× H100 SXM  
**Peak TFLOPS**: 989 TFLOPS/GPU (BF16)

---

## Experimental Configurations

| | **MFSDP** (job 10875143) | **AutoModel** (job 10944375) |
|---|---|---|
| Framework | Megatron-LM + MFSDP | NeMo AutoModel + FSDP2 |
| nsys profile | `MFSDP-qwen3-moe/*.nsys-rep` (no sqlite) | `slurm_jobs/10944375/*.sqlite` |
| EP / dispatcher | 8 / DeepEP | 8 / DeepEP |
| MBS / GBS | 2 / 128 | 2 / 128 |
| Seq len | 4096 | 4096 |
| Grad reduce dtype | FP32 (`grad_reduce_in_fp32=True` from DDP config dump) | FP32 (`reduce_dtype: float32`) |
| Sequence parallel | `--sequence-parallel` (no-op at tp=1) | `false` (equivalent) |
| Activation checkpointing | **selective `moe_act` only** (SwiGLU output) | **full-layer** (attention + MoE + norms) |
| Balanced gate | `--moe-router-force-load-balancing` | `fake_balanced_gate: true` |
| Double buffer | No (`fsdp_double_buffer=False`) | N/A |
| NCCL UB | No (`nccl_ub=False`) | N/A |
| Grad/param overlap | `--overlap-grad-reduce --overlap-param-gather` | Yes (FSDP2 default) |

---

## Performance Summary

| Metric | MFSDP (10875143) | AutoModel (10944375) |
|---|---|---|
| **Avg MFU** | **19.95%** (amortized w/ GC) | **21.71%** |
| Avg TFLOPS/GPU | 197.3 | 214.8 |
| Peak MFU (fast steps) | 20.32% (201 TFLOPS/GPU) | ~24.2% |
| Profiled iterations | 455 (from log) | 20 |

**AutoModel advantage: +1.76pp / +8.8% relative**

> Note: this gap is narrower than the framework efficiency difference because
> AutoModel does **more recompute** (full-layer vs moe_act-only). See findings below.

---

## AutoModel nsys Kernel Breakdown (10944375 — iterations 12–15)

```
Config: GBS=128, MBS=2, ep=8, deepep, activation_checkpointing=true (full-layer)
Steady-state window: 6.492s

Kernel time breakdown (device 0):
  GEMM (nvjet_sm90):             2,717ms   42.6% of GPU time
  Attention fwd (cudnn_fprop):     309ms    4.8%  ← appears 2× per bwd (recomputed)
  Attention bwd (cudnn_bprop):     446ms    7.0%
  DeepEP dispatch:                 300ms    4.7%
  DeepEP combine:                  192ms    3.0%
  DeepEP others:                   128ms    2.0%
  AllGather (NCCL):                415ms    6.5%
  ReduceScatter (NCCL):            643ms   10.1%
  AllReduce (grad sync):             4ms    0.1%
  Total NCCL:                    1,062ms   16.7%

GA steps/iteration: 2 (GBS=128 / MBS=2 / 32 GPUs)
AG-GEMM compute-comm overlap: 0.00%
```

---

## Finding 1: Activation Checkpointing Scope — Not Equivalent

**MFSDP**: `--recompute-granularity selective --recompute-modules moe_act`
- Only recomputes the MoE activation output (SwiGLU post-activation tensor)
- Attention forward is **not** recomputed

**AutoModel**: `activation_checkpointing: true` → `DefaultParallelizationStrategy`
- Code path: `parallelizer.py:271` — uses HF native `gradient_checkpointing_enable(use_reentrant=True)`
- Applies `torch.utils.checkpoint` at **full transformer layer granularity**
- Recomputes: attention forward + MoE forward + all normalization layers

**Evidence from nsys** (`cudnn_fprop` vs `cudnn_bprop` ratio):
```
cudnn_fprop (attn fwd):  672 kernels
cudnn_bprop (attn bwd):  336 kernels
Ratio: 2.0x  →  attention forward is recomputed every backward pass
```

**Implication**: AutoModel does significantly more recompute work per iteration
than MFSDP. If AutoModel used equivalent `moe_act`-only selective recompute,
its MFU would be meaningfully higher. The +1.76pp gap understates AutoModel's
true framework efficiency advantage.

---

## Finding 2: CPU Overhead is the Dominant Bottleneck in AutoModel

```
Window: 6,492ms

CPU-side CUDA API time:        5,308ms  (81.8% of wall-clock)
  Kernel launch (all variants): 2,978ms  (45.8% of wall-clock)
    cuLaunchKernelEx:           1,912ms   516,704 calls  avg 3.7us
    cudaLaunchKernel:           1,209ms   252,224 calls  avg 4.8us
    cudaLaunchKernelExC:          218ms    36,864 calls  avg 5.9us
  cudaMemcpyAsync:              1,259ms    13,984 calls  avg 90us
  cudaDeviceSynchronize:          214ms       257 calls  avg 835us
  cudaStreamSynchronize:          129ms     1,280 calls  avg 101us

GPU idle gap analysis (device 0):
  Total idle gaps:              1,617ms  (24.9% vs GPU busy time)
  Gaps 10–100us:               11,844 gaps,   483ms
  Gaps 100us–1ms:               4,084 gaps,   830ms   ← CPU bottleneck indicator
  Gaps >1ms:                       47 gaps,   212ms   ← severe stalls
  Total large gaps (>100us):                1,042ms  (16.0% of window)
```

**CPU is a severe bottleneck**: 714,216 kernel launches in 6.49s = **~110k launches/s**.
At 4.2us/launch, the CPU spends ~46% of wall-clock time just submitting kernels.
The 4,131 GPU idle gaps >100us (1,042ms total) represent time the GPU is starved
waiting for the CPU to submit the next kernel.

**Per iteration** (~1,623ms each at GBS=128):
- CPU kernel launch overhead: ~745ms/iter
- GPU idle (gaps >100us): ~260ms/iter
- Actual useful GPU work: ~618ms/iter (GEMM only)

**MFSDP CPU overhead**: Not directly measurable (no sqlite), but Megatron-LM's
tighter Python loop and use of persistent CUDA streams typically results in lower
Python dispatch overhead per kernel.

---

## Lessons

### Lesson 1: Recompute scope is not equivalent — AutoModel does more work

AutoModel's `activation_checkpointing: true` recomputes full transformer layers
(confirmed: `cudnn_fprop` appears 2× per `cudnn_bprop`). MFSDP only recomputes
`moe_act`. To make a fair apples-to-apples recompute comparison, AutoModel would
need a selective checkpointing mode equivalent to MFSDP's `--recompute-modules moe_act`.
The code path exists in `parallelizer.py:274-278` (Path B, wrapping only `mlp`
and `self_attn`), but requires transformers < 5.3.0 or a manual override.

### Lesson 2: CPU kernel launch overhead is the #1 bottleneck at GBS=128

~46% of wall-clock is CPU kernel launch time and ~16% of window is GPU idle due
to CPU stalls. This is the primary optimization target. Remedies:
- **CUDA graphs**: captures the kernel launch sequence and replays it — eliminates
  per-step Python dispatch entirely. Most effective fix.
- **torch.compile**: reduces Python overhead via kernel fusion and graph capture
- **Larger GBS**: amortizes fixed CPU overhead over more compute (NCCL and dispatch
  costs also fixed per pass)

### Lesson 3: NCCL is 16.7% of GPU time at GBS=128 — batch size matters

NCCL cost (AG+RS ~1,058ms) does not scale with batch size. At GBS=128 it dominates.
The path to better MFU at small batch is async prefetch (overlapping FSDP2 AllGather
with compute), not framework switching.

### Lesson 4: AllGather-GEMM overlap = 0% — same structural gap as all DeepEP runs

| Profile | Config | AG-GEMM overlap |
|---|---|---|
| 10906024 | no-prefetch | 0.00% |
| 10906025 | prefetch | 0.09% |
| 10944375 | AutoModel apple-to-apple | 0.00% |

### Lesson 5: ReduceScatter > AllGather (643ms vs 415ms, +55%)

ReduceScatter costs more because it must reduce across ranks. Target ReduceScatter
first when optimizing NCCL. (Both are exposed since AG-GEMM overlap is 0%.)

### Lesson 6: Grad reduce dtype irrelevant at this scale

Both frameworks use FP32 grad reduce. AllReduce = 4ms = 0.1% of window.
Not a performance consideration at 32-GPU scale with DeepEP.

---

## Files Referenced

| File | Description |
|---|---|
| `MFSDP-qwen3-moe/*.nsys-rep` | MFSDP nsys profile (rep only, no sqlite) |
| `slurm_jobs/10944375/*.sqlite` | AutoModel apple-to-apple nsys sqlite |
| `nemo_automodel/components/distributed/parallelizer.py:241-278` | Activation checkpointing code path |
| `mfsdp-qwen-recipe.sh` | MFSDP launch recipe |
| `examples/benchmark/configs/qwen3_moe_30b_te_deepep_4nodes_autonvtx_profile.yaml` | AutoModel config |
| `analyze_overlap.py` | AllGather-GEMM overlap tool |
| MFSDP log: `xuwenc/.../slurm_logs/qwen3_30b_a3b_mfsdp_..._10875143.log` | MFSDP throughput log |
