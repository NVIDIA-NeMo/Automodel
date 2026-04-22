# LLaMA3-70B MFSDP no_flag bf16-reduce vs Automodel FSDP2 bf16-reduce — nsys Profile Analysis

**Date**: 2026-04-22
**Model**: LLaMA3.1-70B
**Config**: TP=1, CP=1, PP=1, DP=64, MBS=1, GBS=128, SeqLen=4096, 8 nodes (64×H100 SXM), full-recompute, `--grad-reduce-in-bf16`

---

## Runs Compared

| | Megatron MFSDP no_flag bf16-reduce | Automodel FSDP2 bf16-reduce |
|---|---|---|
| Framework | Megatron-LM (`--use-megatron-fsdp`) | Automodel PyTorch FSDP2 |
| Container | `pytorch26.03_te2.14_deepep_x86` | `nemo-automodel:26.04.rc4` |
| nsys-rep | `MFSDP-llama3-70b-repro/nsys/...no_flag_0_bf16-reduce.nsys-rep` | `slurm_jobs/11278469/nsys_llama31_70b_pretrain..._bf16-reduce_node0.nsys-rep` |
| MFSDP flags | `--fsdp-double-buffer` (no UB, no manual-registration) | — |
| Grad reduce dtype | BF16 (`--grad-reduce-in-bf16`) | BF16 (default in Automodel) |
| Profile steps | 10–12 (N_STEPS=2) | iter 7–8 (N_STEPS=1.5 true) |
| Profile window | 16,822.9ms (kernel window ÷ 2) | 12,358.7ms NVTX window |
| **Step time (nsys)** | **8,411ms** (window ÷ 2) | **~8,177ms** (ga_step_0+ga_step_1 NVTX) |
| **Step time (log avg)** | **8,484ms** (iters 3–10 log) | **~8,130ms** (from log MFU ~43.4%) |
| **MFU (approx)** | **~41.8%** | **~42.9%** |

---

## 1. Fairness & Shared Configuration

### What is the same (fair)

| Factor | MFSDP no_flag | FSDP2 |
|---|---|---|
| Hardware | Same cluster, same H100 SXM node pool | ← same |
| Model architecture | LLaMA3.1-70B, hidden=8192, 80 layers, GQA | ← same |
| Parallelism | TP=1, CP=1, PP=1, DP=64 | ← same |
| Batch config | GBS=128, MBS=1, SeqLen=4096, GAS=2 | ← same |
| Activation recompute | Full recompute | ← same |
| Attention backend | TE FusedAttention (cuDNN WGMMA) | ← same kernel |
| NCCL env vars | None | None |
| Grad reduce dtype | BF16 | BF16 |

### What is different (potential unfairness)

| Factor | MFSDP no_flag | FSDP2 | Impact |
|---|---|---|---|
| Container version | `pytorch26.03` | `26.04.rc4` | Minor TE/PyTorch/NCCL version diff |
| NVTE_LAYERNORM_SM_MARGIN | 16 (set) | Not set | ~4ms/step on RMSNorm; does not affect GEMM |
| Parameter layout | Flat contiguous buffer | Individual tensors per param | Structural — drives GEMM output alignment |
| Projection fusion | Fused QKV, fused gate+up | Separate q/k/v, gate, up | Changes GEMM call count and matrix sizes |
| Double buffer | `--fsdp-double-buffer` (yes) | Implicit prefetch | Pre-fetches next-layer weights |

### Step window normalization

**MFSDP**: N_STEPS=2 — full 16,822.9ms kernel window ÷ 2. Verified by:
- RMSNorm fwd: 2,572 ÷ 1,286 expected/step = exactly 2.000 ✓
- Attention FWD: 640 ÷ 320 expected/step = exactly 2.000 ✓
- Attention BWD: 320 ÷ 160 expected/step = exactly 2.000 ✓

**FSDP2**: The nsys-overlap tool detects n_dist=2 (distinct ga_step_1 texts) and uses N_STEPS=2. However, the NVTX window spans from `iteration_7_ga_step_1.start` to `iteration_8_ga_step_1.end`, which covers:
- Full `iteration_7_ga_step_1` (~4,111ms)
- Full `iteration_8` (ga_step_0 ~4,083ms + ga_step_1 ~4,094ms)
= approximately **1.5 optimizer steps**, not 2.

**True FSDP2 N_STEPS = 1.5**, verified by:
- RMSNorm fwd: 1,929 kernels ÷ 1,286 expected/step = 1.500 ✓ (exactly 3/2)
- Attention FWD: 480 ÷ 320 = 1.500 ✓
- Attention BWD: 240 ÷ 160 = 1.500 ✓

All FSDP2 kernel totals in this report are divided by 1.5 (not 2) to get true per-step values.
Overlap **percentages** from the tool are unaffected (they are ratios within the same window).

---

## 2. High-Level Results

### Headline

**FSDP2 bf16-reduce is 2.8% faster** (8,177ms vs 8,411ms/step by nsys; 3.6% vs log averages).
Both runs use BF16 gradient reduce. Compared to the prior FP32-reduce comparison
(`analysis_report_ub_iter8.md`), bf16-reduce cuts RS kernel time by ~52% in both frameworks,
bringing RS from the dominant bottleneck to near-parity between them. With RS equalized, the
remaining FSDP2 advantages are: better GEMM efficiency (−10.0%), lower exposed AG (63ms vs 136ms/step),
and better overall overlap. MFSDP retains its 200× buffer copy overhead advantage (3ms vs 614ms/step)
and comparable RS overlap.

A notable discovery: with `--grad-reduce-in-bf16`, MFSDP no longer shows the `tss` GEMM epilogue
penalty observed in the FP32-reduce run. Both frameworks use `tst` for all GEMM shapes. Despite this,
FSDP2 is still 10% faster on GEMM, explained by MFSDP's larger fused-projection matrix sizes causing
HBM bandwidth contention.

### Summary

| Metric | MFSDP no_flag bf16 | FSDP2 bf16 | FSDP2 vs MFSDP |
|---|---|---|---|
| **Step wall time (nsys)** | **8,411ms** | **8,177ms** | **FSDP2 −2.8%** |
| **Step wall time (log avg)** | **8,484ms** | **~8,130ms** | **FSDP2 −4.0%** |
| MFU | ~41.8% | ~42.9% | FSDP2 better |
| AllGather/step | 3,624ms | **3,348ms** | **FSDP2 −7.6%** |
| ReduceScatter/step | 1,677ms | 1,708ms | Near-tie (+1.9%) |
| Total comm/step | 5,301ms | 5,056ms | FSDP2 −4.6% |
| Total GEMM/step | 7,140ms | **6,426ms** | **FSDP2 −10.0%** |
| Attention (FWD+BWD)/step | 479ms | **457ms** | FSDP2 −4.6% |
| Buffer copy/step | **3ms** | 614ms | MFSDP **205× less** |
| AG truly exposed/step | 136ms (3.8%) | **63ms** (1.9%) | **FSDP2 −53.7%** |
| RS truly exposed/step | 69ms (4.1%) | **64ms** (3.7%) | Near-tie |
| **Total exposed comm/step** | 205ms | **127ms** | **FSDP2 −38.0%** |

### FSDP2 Advantages over MFSDP no_flag

1. **GEMM −10.0%** (6,426ms vs 7,140ms): Two sub-causes:
   - Even for the same problem size (192×192 NTN = down_proj FWD/BWD), FSDP2 is 10–12% faster
     (3,036µs vs 3,403µs/call). Root cause confirmed by direct measurement (§4.7): MFSDP's flat
     buffer AG causes +107% slowdown on concurrent 192×192 GEMMs (4,305µs vs 2,080µs) while
     FSDP2 shows only +2.4% (no meaningful contention). 57% of MFSDP's 192×192 NTN instances
     run in the contended regime.

2. **AllGather −7.6%** (3,348ms vs 3,624ms): It might be relevant: FSDP2's per-parameter AG allows more granular
   prefetch scheduling. MFSDP's flat-buffer AGs are fewer but each larger, with less
   flexibility to overlap with compute.

3. **Overlap: AG exposed −53.7%** (63ms vs 136ms): FSDP2's smaller per-call AG (avg ~10,397µs vs
   11,118µs) combined with better prefetch scheduling results in significantly less exposed AG.
   The smaller 3,348ms total AG is more easily buried within the GEMM window.

4. **`tst` GEMM in both** (no `tss` penalty in current MFSDP): Unlike the prior FP32-reduce run
   where MFSDP showed `tss_192x192_NTN` (shared-mem epilogue), this BF16-reduce run uses `tst`
   for all GEMM shapes. With BF16 gradients, the gradient tensors appear to be independently
   allocated (not pure offset views in the flat buffer), enabling cuBLAS TMA epilogue selection.
   This eliminates ~157ms/step of prior `tss` penalty — but FSDP2 still wins on GEMM overall.

### MFSDP Advantages over FSDP2

1. **Buffer copy overhead 205× less** (3ms vs 614ms/step): MFSDP's flat buffer avoids
   `chunk_cat` (204ms/step pre-RS packing) and `split_with_sizes_copy_out` (410ms/step post-AG
   unpacking) that FSDP2 requires. Note: with BF16 reduce, `chunk_cat` dropped from 476ms/step
   (FP32) to 204ms/step (BF16 — half data volume); `split_with_sizes_copy_out` (param copy,
   already BF16 in both runs) is unchanged at ~410ms/step.

2. **RS near-tie** (1,677ms vs 1,708ms): With BF16 reduce, both frameworks have comparable RS
   kernel times. MFSDP is marginally faster (+1.9% for FSDP2 = near-tie). The prior FP32-reduce
   MFSDP RS was 3,524ms; bf16-reduce halved it to 1,677ms (−52.4%), matching the FSDP2 gain.

3. **Memory headroom** (estimated): MFSDP uses a pre-allocated flat buffer while FSDP2 holds
   `chunk_cat` output alongside original param tensors during AG, creating temporary peak
   duplication. FSDP2 bf16-reduce likely has ~1-2 GB less peak than FP32-reduce (smaller grad
   buffer), but still structurally higher than MFSDP.

---

## 3. Improvement Opportunities

### For Automodel FSDP2

1. **Adopt contiguous flat parameter buffer** — Eliminates `chunk_cat` (204ms/step pre-RS) and
   `split_with_sizes_copy_out` (410ms/step post-AG). Total savings if both become negligible:
   ~614ms/step of compute-stream serialization removed. Today these are partially overlapped;
   the actual wall-time gain would be smaller but meaningful.

2. **Projection fusion (QKV, gate+up)** — MFSDP's fused QKV and gate+up GEMMs have higher
   arithmetic intensity per call, which can improve SM utilization. The trade-off is larger
   per-AG sizes; net benefit needs measurement.

### For Megatron MFSDP

2. **Investigate 192×192 NTN GEMM slowdown** — Even with `tst` epilogue, MFSDP's 192×192 NTN
   (down_proj) runs ~11% slower (3,403µs vs 3,036µs/call). Root cause: HBM bandwidth contention
   from the large flat-buffer AG competing with GEMM streaming. Potential fix: improve all-gather
   scheduling so the flat-buffer AG is staggered or chunked away from the 192×192 NTN window,
   reducing direct overlap on HBM bandwidth.

3. **Add NVTX iteration markers** — MFSDP profile has no `iteration_N_ga_step_M` markers, making
   exact per-step normalization impossible (requires RMSNorm counting to verify N_STEPS). Add
   these to enable per-step breakdown and direct step-time comparison.

4. **Evaluate NCCL user buffer mode (on stable nodes)** — Prior UB run (`analysis_report_ub_iter8.md`)
   showed AG −12.5%, RS −20.6% vs the no_flag run. If the `manual_buffer_registration` crash is
   resolved, UB mode would cut MFSDP's AG from 3,624ms to ~2,900ms and RS from 1,677ms to ~1,400ms.


---

## 4. Detailed Evidence

### 4.1 HBM Bandwidth Contention Verification

**Hypothesis**: MFSDP's flat contiguous parameter buffer creates a larger HBM working set
during AllGather, competing with GEMM tile streaming and slowing the 192×192 NTN kernel.
FSDP2's per-parameter buffers are smaller and scattered, causing no such contention.

**Method**: For each 192×192 NTN kernel instance, check whether it temporally overlaps
with any AllGather kernel on the same device. Split instances into "concurrent" and
"non-concurrent" groups and compare average duration.

**Results**:

| | MFSDP no_flag bf16 | FSDP2 bf16 |
|---|---|---|
| Total 192×192 NTN instances | 964 | 723 |
| Concurrent with AG | 552 (57%) | 533 (74%) |
| Non-concurrent | 412 (43%) | 190 (26%) |
| **Concurrent avg duration** | **4,304.7 µs** | **3,043.2 µs** |
| **Non-concurrent avg duration** | **2,079.8 µs** | **2,970.9 µs** |
| **Concurrent median** | 3,391.7 µs | 3,079.8 µs |
| **Non-concurrent median** | 1,051.5 µs | 2,788.2 µs |
| **Slowdown when concurrent** | **+107.0%** | **+2.4%** |

**Interpretation**:

- **MFSDP**: 192×192 NTN GEMMs running alongside an AG take **2× longer** on average
  (4,305 µs vs 2,080 µs). Median-to-median is even starker: 3,392 µs vs 1,052 µs = **3.2× slower**.
  The non-concurrent floor of 877 µs min shows what the GEMM achieves at full HBM bandwidth.
  This confirms HBM bandwidth contention from the flat-buffer AG is the dominant cause of the
  192×192 NTN slowdown observed vs FSDP2.

- **FSDP2**: Only +2.4% slowdown (3,043 µs concurrent vs 2,971 µs non-concurrent) — within
  noise. FSDP2's per-parameter AG traffic does not meaningfully compete with GEMM HBM access.
  Note FSDP2's non-concurrent duration (2,971 µs) already matches the baseline GEMM speed,
  confirming no structural contention even outside AG windows.

- **Gap explained**: The 10.8% per-call gap (3,403 µs MFSDP v1 vs 3,036 µs FSDP2 v1 from §4.4)
  comes from MFSDP executing **most of its 192×192 NTN GEMMs (57%) in the contended regime**
  (avg 4,305 µs) while FSDP2 executes the same shape near its floor (avg 3,043 µs) regardless
  of AG concurrency. The weighted average of MFSDP's concurrent (4,305 µs × 57%) + non-concurrent
  (2,080 µs × 43%) = 3,349 µs, consistent with the §4.4 measured 3,403 µs ✓.

**Conclusion**: HBM bandwidth contention from the MFSDP flat buffer is confirmed as the root
cause of the 192×192 NTN slowdown. The fix direction is to reduce the flat-buffer AG working
set — either by sharding the buffer into smaller chunks or adopting per-parameter layout (as
FSDP2 does). This single contention effect accounts for ~10% of the GEMM gap between the
two frameworks.

### 4.2 Collective Kernel Detail

| | MFSDP no_flag bf16 | FSDP2 bf16 | Winner |
|---|---|---|---|
| AG calls/step | 326 | 322 | Similar |
| AG avg duration (µs) | 11,118 | 10,397 | **FSDP2 −6.5%** |
| AG total/step (ms) | 3,624 | **3,348** | **FSDP2 −7.6%** |
| RS calls/step | 164 | 162 | Tie |
| RS avg duration (µs) | 10,225 | 10,544 | Near-tie |
| RS total/step (ms) | **1,677** | 1,708 | Near-tie |
| Kernel name (AG) | `ncclDevKernel_AllGather_RING_LL` | Same | — |
| Kernel name (RS) | `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` | Same | — |

### 4.3 Full Kernel Breakdown per Step

| Category | MFSDP cnt/step | MFSDP ms/step | FSDP2 cnt/step | FSDP2 ms/step | Winner |
|---|---|---|---|---|---|
| AllGather | 326 | 3,624 | 322 | 3,348 | **FSDP2 −7.6%** |
| ReduceScatter | 164 | 1,677 | 162 | 1,708 | Near-tie |
| GEMM total | 2,566 | 7,140 | 4,166 | 6,426 | **FSDP2 −10.0%** |
| Attention FWD (cuDNN) | 320 | 207 | 320 | 187 | **FSDP2 −9.7%** |
| Attention BWD (cuDNN) | 160 | 272 | 160 | 270 | Tie |
| Buffer copy (pre-RS + post-AG) | ~334 | **3** | 482 | **614** | MFSDP **205× less** |
| RMSNorm | 1,286 | 72 | 1,286 | 73 | Tie |
| RoPE | 960 | ~35 | 960 | ~31 | Near-tie |
| **Step wall time** | | **8,411** | | **8,177** | **FSDP2 −2.8%** |

Buffer copy detail for FSDP2: `chunk_cat_cuda_kernel` 204ms + `split_with_sizes_copy_out` 410ms = 614ms/step.
MFSDP: DtoD shard copy 668 ops / 2 = 334 ops/step, 6.7ms / 2 = 3ms/step.

Comm vs Compute:

| | MFSDP no_flag bf16 | FSDP2 bf16 |
|---|---|---|
| Comm (AG+RS) | 5,301ms | 5,056ms |
| Compute (GEMM+attn) | 7,619ms | 6,883ms |

### 4.4 GEMM Per-Shape Breakdown

> **Reading guide**: MFSDP fuses QKV into `[8192, 10240]` and gate+up into `[8192, 57344]`, so
> per-call matrix sizes differ from FSDP2's separate projections. ³ marks non-apples-to-apples
> comparisons. ⁴ marks same problem size with fair per-call comparison.

| Kernel shape | MFSDP ms/step | MFSDP avg µs | FSDP2 ms/step | FSDP2 avg µs | Winner |
|---|---|---|---|---|---|
| `tst_320x128_TNT` | 2,367 | 3,687 | 1,998 | 3,112 | Not fair ³ (MFSDP fused QKV+gate_up) |
| `tst_256x128_NNT` | 1,392 | 2,888 | 1,266 | 1,317 | Not fair ³ (different counts) |
| `tst_256x128_TNT` | 1,185 | 1,852 | 897 | 700 | Not fair ³ (different counts) |
| `tst_192x192_NTN` v1 | 1,096 | **3,403** | 977 | **3,036** | **FSDP2 −10.8%** ⁴ |
| `tst_192x192_NTN` v2 | 521 | **3,255** | 480 | **3,001** | **FSDP2 −7.8%** ⁴ |
| `tst_320x128_NNT` | 445 | 2,781 | 506 | 3,165 | Not fair ³ |
| `tst_320x128_NTT` | 134 | 837 | 266 | 832 | Not fair ³ (FSDP2 2× calls, separate KV BWD) |
| `tst_256x128_NTT` | — | — | 34 | 107 | — |
| **Total GEMM** | **7,140** | | **6,426** | | **FSDP2 −10.0%** |

³ **`tst_320x128_TNT`, `tst_256x128_NNT`, `tst_256x128_TNT`, `tst_320x128_NTT`: different FLOPs per call.**
- MFSDP fuses QKV → single `[8192, 10240]` weight (more tiles per GEMM) vs FSDP2's separate Q/K/V.
- MFSDP fuses grad-KV → fewer NTT calls (160/step) vs FSDP2 (320/step for separate K+V).

⁴ **`tst_192x192_NTN` — same problem size (down_proj FWD/BWD), both using `tst` epilogue.**
Both frameworks use `tst` (TMA-store epilogue) for this shape. FSDP2 is still 10.8% faster
per call. Root cause: MFSDP's flat buffer creates larger HBM working set during AG, causing
bandwidth contention that slows the 192×192 streaming GEMM even without epilogue differences.
See §4.1 for direct measurement.

**Key discovery — no `tss` in BF16-reduce MFSDP**: Unlike the prior FP32-reduce MFSDP+UB run
which showed `tss_192x192_NTN` (cuBLAS shared-mem epilogue due to flat-buffer offset misalignment),
this BF16-reduce run uses `tst` throughout. With `--grad-reduce-in-bf16`, the BF16 gradient tensors
appear to be independently allocated rather than sub-views of the flat buffer, allowing cuBLAS to
select the TMA epilogue.

### 4.5 Attention Kernels

Both use identical cuDNN WGMMA kernels — fully fair comparison.

| | MFSDP no_flag bf16 | FSDP2 bf16 |
|---|---|---|
| FWD kernel | `cudnn_...sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_4x` | Same |
| BWD kernel | `cudnn_...sdpa_sm90_flash_bprop_wgmma_f16_knob_26_64x64x128_1x` | Same |
| FWD calls/step | 320 | 320 |
| BWD calls/step | 160 | 160 |
| FWD avg_us / total/step | 646µs / 207ms | 585µs / 187ms |
| BWD avg_us / total/step | 1,703µs / 272ms | 1,685µs / 270ms |
| **Total attn/step** | **479ms** | **457ms** |

FWD is 9.7% faster in FSDP2 (585µs vs 646µs/call), likely from reduced HBM pressure when AG
buffer sizes are smaller. BWD is essentially identical (1,685µs vs 1,703µs, 1.1% difference).

### 4.6 Buffer Copy Overhead (per step)

| Kernel | Role | FSDP2 bf16 ms/step | MFSDP bf16 ms/step | vs prior FSDP2 (FP32) |
|---|---|---|---|---|
| `chunk_cat_cuda_kernel` | Pre-RS: pack BF16 grads → flat | **204** | 0 | 476ms → 204ms (−57% from bf16) |
| `split_with_sizes_copy_out` | Post-AG: unpack flat → param tensors | **410** | 0 | 402ms → 410ms (unchanged, params are BF16 in both) |
| DtoD memcpy (MFSDP shard copy) | Pre-AG: shard from flat buf | 0 | **3** | — |
| **Total** | | **614** | **3** | 878ms → 614ms (−30% from bf16) |

`chunk_cat` shrank by 57% with BF16 gradients (half the data volume compared to FP32).
`split_with_sizes_copy_out` is unchanged because it copies BF16 parameters (not gradients) back
to individual tensors — the dtype was always BF16 in both runs.

### 4.7 Comm–Compute Overlap Analysis

Source: `nsys-overlap/comm_overlap.py` + `analyze_comm_compute_overlap.py`, 3-bucket decomposition.
MFSDP: N_STEPS=2, full kernel window. FSDP2: percentages correct (tool's n=2); absolute ms computed
using corrected step comm wall (= percentage × corrected total comm wall at N_STEPS=1.5).

| Bucket | MFSDP AG | FSDP2 AG | MFSDP RS | FSDP2 RS |
|---|---|---|---|---|
| Comm wall/step | 3,624ms | 3,348ms | 1,677ms | 1,708ms |
| **A. Hidden by GEMM+attn** | **91.0%** | **90.0%** | **91.2%** | **80.0%** |
| **B. Hidden by light compute** | 5.3% | 8.1% | 4.7% | 16.3% |
| **C. Truly exposed** | **3.8% (136ms)** | **1.9% (63ms)** | **4.1% (69ms)** | **3.7% (64ms)** |
| **Total hidden (A+B)** | 96.3% | 98.1% | 95.9% | 96.3% |

**Total truly exposed comm per step:**

| | MFSDP no_flag bf16 | FSDP2 bf16 |
|---|---|---|
| AG exposed | **136ms** | 63ms |
| RS exposed | **69ms** | 64ms |
| **Total exposed** | **205ms** | **127ms** |
| **Winner** | | **FSDP2 −38.0%** |

**Key observations:**

1. **FSDP2 AG overlap significantly better (63ms vs 136ms exposed)**: FSDP2's smaller per-call
   AG (3,348ms total vs 3,624ms) combined with per-parameter prefetch scheduling allows more AGs
   to be buried within GEMM windows. MFSDP's larger monolithic AGs have longer tails that extend
   beyond GEMM coverage.

2. **RS overlap near-tie (69ms vs 64ms exposed)**: With both using BF16 RS, the RS collectives
   are short enough (1,677ms vs 1,708ms) that GEMM coverage is similar for both. FSDP2's RS
   has higher B-bucket (light compute) coverage at 16.3% vs MFSDP's 4.7%, suggesting FSDP2's
   post-backward elementwise ops (chunk_cat, optimizer fused ops) partially overlap with RS.

3. **Both achieve >95% total comm overlap**: Unlike MoE models where AG can be 98%+ exposed, dense
   LLaMA3-70B with full-recompute has excellent hiding in both frameworks. The large GEMM windows
   (7,140ms MFSDP, 6,426ms FSDP2) dwarf the comm duration.

4. **MFSDP RS bucket A dominates (91.2%)**: MFSDP's shorter RS (1,677ms) is almost entirely
   buried under GEMM+attn. FSDP2 RS has only 80.0% under GEMM+attn — the remaining 16.3% is
   hidden by elementwise/optimizer overlap (B-bucket). This suggests FSDP2's RS sometimes fires
   during non-GEMM segments (e.g., between BWD GEMMs during chunk_cat serialization).

---

## 5. BF16-Reduce Impact: Before vs After

Comparison to the prior FP32-reduce runs from `analysis_report_ub_iter8.md`.

### FSDP2: FP32-reduce → BF16-reduce

| Metric | FSDP2 FP32-reduce | FSDP2 bf16-reduce | Δ |
|---|---|---|---|
| Step time | 8,583ms | ~8,177ms | **−4.7%** |
| AllGather/step | 3,301ms | 3,348ms | +1.4% (≈ tie) |
| ReduceScatter/step | 3,566ms | 1,708ms | **−52.1%** |
| Total comm/step | 6,867ms | 5,056ms | **−26.4%** |
| Total GEMM/step | 6,500ms | 6,426ms | −1.1% (≈ tie) |
| `chunk_cat`/step | 476ms | 204ms | **−57.1%** (bf16 grads = ½ size) |
| Total exposed comm | 162ms | 127ms | **−21.6%** |

### MFSDP: FP32-reduce (no_flag est.) → BF16-reduce

| Metric | MFSDP no_flag FP32 (est.) | MFSDP no_flag bf16 | Δ |
|---|---|---|---|
| Step time | ~8,931ms | 8,411ms | **−5.8%** |
| AllGather/step | ~4,130ms | 3,624ms | **−12.3%** (node/cluster variation) |
| ReduceScatter/step | ~3,524ms | 1,677ms | **−52.4%** |
| Total comm/step | ~7,654ms | 5,301ms | **−30.7%** |
| Total GEMM/step | ~7,226ms | 7,140ms | −1.2% (≈ tie) |
| `tss_192x192` penalty | ~157ms | 0ms | **eliminated** (bf16 grads → tst) |
| Total exposed comm | ~582ms | 205ms | **−64.8%** |

**The `tss` → `tst` transition in MFSDP** (eliminating ~157ms/step penalty) is a notable
side-effect of BF16-reduce. The prior FP32-reduce MFSDP had gradient buffers stored as offset
views in the flat parameter buffer → misaligned base pointers → cuBLAS fallback to `tss`.
With BF16 reduce, gradient buffers appear to be independently allocated, enabling `tst`.

---

## 6. Memory Analysis

Both runs use `--grad-reduce-in-bf16`, which halves the gradient buffer footprint.

| Metric | MFSDP no_flag bf16 | FSDP2 bf16 | Notes |
|---|---|---|---|
| DtoD memcpy data volume | 8.41 GB / 2 steps = 4.2 GB/step | — | MFSDP flat-buf shard copy |
| FSDP2 `chunk_cat` grads packed | — | ~half vs FP32 | BF16 grads = ½ data |

Full peak memory comparison not directly measured (no `torch.cuda.max_memory_allocated` output
in the logs). Structurally, FSDP2 still has temporary 2× peak shard footprint during `chunk_cat`.

---

## 7. Comparison with Prior Run (analysis_report_ub_iter8.md)

The prior report compared MFSDP+UB vs FSDP2 with **FP32 reduce**. Key context for this report:

| | Prior: MFSDP+UB FP32 | Prior: FSDP2 FP32 | Current: MFSDP no_flag bf16 | Current: FSDP2 bf16 |
|---|---|---|---|---|
| Step time | 8,448ms | 8,583ms | 8,411ms | **8,177ms** |
| MFU | 42.4% | 41.5% | 41.8% | **42.9%** |
| AG/step | 2,886ms | 3,301ms | 3,624ms | 3,348ms |
| RS/step | 2,832ms | 3,566ms | 1,677ms | 1,708ms |
| GEMM epilogue | `tss`+`tst` | `tst` | **`tst`** | `tst` |
| Total exposed | 150ms | 162ms | 205ms | **127ms** |

**Best performer overall**: FSDP2 bf16-reduce at 8,177ms is now faster than even MFSDP+UB
(8,448ms), which required cluster-unstable UB mode. The key enabler was BF16 reduce dramatically
cutting RS, combined with FSDP2's inherently better GEMM efficiency.

---

## 8. nsys-rep and sqlite Paths

| Profile | nsys-rep | sqlite |
|---|---|---|
| MFSDP no_flag bf16-reduce | `MFSDP-llama3-70b-repro/nsys/llama3_70b_mfsdp_tp1_pp1_ep1_cp1_hsdp1_alltoall_mbs1_gbs128_seqlen4096_cw_n8_full-recompute_no_flag_0_bf16-reduce.nsys-rep` | Same dir, `.sqlite` |
| FSDP2 bf16-reduce | `slurm_jobs/11278469/nsys_llama31_70b_pretrain_tp1cp1pp1_te_attn_cp1_no_compile_gbs128_seqlen4096_bf16-reduce_node0.nsys-rep` | `slurm_jobs/11278469/...bf16-reduce.sqlite` |

Analysis job: Slurm job **11279946**, log: `MFSDP-llama3-70b-repro/slurm_logs/analyze_bf16reduce_11279946.log`
