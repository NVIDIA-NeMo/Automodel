# LLaMA3-70B MFSDP+UB vs Automodel FSDP2 — nsys Profile Analysis (CP=1, iter-8-exact)

**Date**: 2026-04-15
**Model**: LLaMA3.1-70B
**Config**: TP=1, CP=1, PP=1, DP=64, MBS=1, GBS=128, SeqLen=4096, 8 nodes (64×H100 SXM), full-recompute

---

## Runs Compared

| | Megatron MFSDP+UB | Automodel FSDP2 |
|---|---|---|
| Framework | Megatron-LM (`--use-megatron-fsdp`) | Automodel PyTorch FSDP2 |
| Container | `pytorch26.03_te2.14_deepep_x86` | `nemo-automodel:26.04.rc4` |
| nsys-rep | `MFSDP-llama3-70b-repro/llama3_70b_mfsdp_tp1_pp1_ep1_cp1_hsdp1_alltoall_mbs1_gbs128_seqlen4096_cw_n8_full-recompute_no_flag_0_ub.nsys-rep` | `slurm_jobs/11124101/nsys_llama31_70b_pretrain_tp1cp1pp1_te_attn_cp1_no_compile_gbs128_seqlen4096_node0.nsys-rep` |
| Extra MFSDP flags | `--fsdp-double-buffer --use-nccl-ub --fsdp-manual-registration` | — |
| NCCL env vars | None | None |
| Profile steps | 10–11 | 7–8 |
| Profile window | 16,897ms (**N_STEPS=2**, no NVTX — kernel-count verified) | **Exact iter 8 NVTX window** (8,583ms, N_STEPS=1) |
| **Step time** | **8,448ms** (nsys ÷ 2) | **8,583ms** (NVTX iter 8 window); log avg **8,646ms** |
| **MFU (approx)** | **~42.4%** | **~41.5%** |

---

## 1. Fairness & Shared Configuration

### What is the same (fair)

| Factor | MFSDP+UB | FSDP2 |
|---|---|---|
| Hardware | Same cluster, same H100 SXM node pool | ← same |
| Model architecture | LLaMA3.1-70B, hidden=8192, 80 layers, GQA | ← same |
| Parallelism | TP=1, CP=1, PP=1, DP=64 | ← same |
| Batch config | GBS=128, MBS=1, SeqLen=4096, GAS=2 | ← same |
| Activation recompute | Full recompute | ← same |
| Attention backend | TE FusedAttention (cuDNN WGMMA) | ← same kernel |
| NCCL env vars | None | None |
| Optimizer dtype | BF16 params, FP32 reduce | ← same |

### What is different (potential unfairness)

| Factor | MFSDP+UB | FSDP2 | Impact |
|---|---|---|---|
| Container version | `pytorch26.03` | `26.04.rc4` | Minor TE/PyTorch/NCCL version diff |
| NVTE_LAYERNORM_SM_MARGIN | 16 (set) | Not set | ~4ms/step on RMSNorm; does not affect GEMM |
| Parameter layout | Flat contiguous buffer | Individual tensors per param | Structural — drives `tss` vs `tst` GEMM variant |
| Projection fusion | Fused QKV, fused gate+up | Separate q/k/v, gate, up | Changes GEMM call count and tile shapes |
| NCCL UB mode | `--use-nccl-ub --fsdp-manual-registration` | Not applicable | Enables NCCL user buffer registration for collectives |
| Double buffer | `--fsdp-double-buffer` | Implicit prefetch | Pre-fetches next-layer weights into a second buffer |

### Window normalization

**MFSDP+UB**: N_STEPS=2 — full 16,897ms kernel window ÷ 2. Confirmed by:
- AG calls: 652 ÷ 326 expected/step = exactly 2.000
- RMSNorm: 2,572 ÷ 1,286 expected/step = exactly 2.000
- Attention FWD: 640 ÷ 320 expected/step = exactly 2.000

**FSDP2**: Kernels queried directly from the **exact iter 8 NVTX window** — no N_STEPS division needed.
NVTX timestamps from `slurm_jobs/11124101/*.sqlite`:
```
iteration_8_ga_step_0: start = 8,649,757,917 ns  (8 ranks captured)
iteration_8_ga_step_1: end   = 17,233,070,459 ns  (8 ranks captured)
Window duration: 8,583.3 ms  = 1 complete optimizer step (2 GA steps)
```

**Verification — RMSNorm count (iter 8 exact):**

| | FSDP2 iter 8 (device 0) | MFSDP+UB per-step (÷ 2) |
|---|---|---|
| Total RMSNorm (all types) | **1,286** | **1,286** |

Exact match → confirms iter 8 = exactly 1 optimizer step. Also, attention FWD calls: FSDP2 iter 8 =
320, MFSDP+UB/step = 320 ✓; attention BWD: both 160 ✓.

---

## 2. High-Level Results & Adv / Disadv

### Headline

**MFSDP+UB is 2.0% faster** (8,448ms vs 8,583ms/step by NVTX; 2.1% vs log 8,646ms). NCCL user buffer
mode gives a meaningful comm kernel advantage (AG −12.5%, RS −20.6%) with negligible buffer copy
overhead (9ms vs 878ms). Both runs achieve ~97% comm-compute overlap — the overlap advantage
for UB over FSDP2 is modest (total exposed: 150ms vs 162ms). The speed advantage comes primarily
from lower comm kernel time and eliminated buffer copy.

### Summary

| Metric | MFSDP+UB (÷ 2) | FSDP2 (iter 8 exact) | UB vs FSDP2 |
|---|---|---|---|
| **Step wall time** | **8,448ms** | 8,583ms | **UB −2.0%** |
| MFU | ~42.4% | ~41.5% | UB better |
| AllGather/step | **2,886ms** | 3,301ms | **UB −12.5%** |
| ReduceScatter/step | **2,832ms** | 3,566ms | **UB −20.6%** |
| Total comm/step | **5,718ms** | 6,867ms | **UB −16.7%** |
| Total GEMM/step | 7,251ms | **6,500ms** | UB +11.5% worse |
| Attention (FWD+BWD)/step | 473ms | **461ms** | Near-tie (FSDP2 −2.5%) |
| Buffer copy overhead/step | **9ms** | 878ms | UB **97× less** |
| AG truly exposed/step | **61ms** (2.1%) | 66ms (2.0%) | Near-tie |
| RS truly exposed/step | **89ms** (3.1%) | 96ms (2.7%) | Near-tie |
| **Total exposed comm/step** | **150ms** | 162ms | UB −7.4% |

### MFSDP+UB Advantages over FSDP2

1. **AllGather −12.5%** (2,886ms vs 3,301ms): NCCL user buffer mode registers parameter buffers
   directly with the NCCL library, enabling zero-copy AG collectives. avg per call: 8,851µs (UB)
   vs 10,316µs (FSDP2).

2. **ReduceScatter −20.6%** (2,832ms vs 3,566ms): UB mode similarly benefits RS by eliminating
   intermediate copies. avg per call: 17,267µs (UB) vs 22,013µs (FSDP2).

3. **Buffer copy overhead 97× less** (9ms vs 878ms/step): MFSDP's flat contiguous buffer eliminates
   `chunk_cat` (476ms/step) and `split_with_sizes_copy_out` (402ms/step) that FSDP2 requires.

4. **Good overlap maintained**: Both runs are ~97% overlap. UB exposed total = 150ms vs FSDP2 162ms
   — a 12ms/step advantage, modest compared to the 1,149ms comm kernel time savings.

### FSDP2 Advantages

1. **GEMM −10.4%** (6,500ms vs 7,251ms/step): The `tss` epilogue penalty persists in MFSDP+UB.
   NCCL UB mode does not affect output tensor memory layout. The flat contiguous buffer still
   produces non-aligned base pointer offsets → cuBLAS selects `tss` (shared-memory epilogue) for
   192×192 FFN tiles instead of `tst` (TMA-store epilogue). `tss` totals ~1,642ms/step (§4.3).

2. **Attention slightly faster** (461ms vs 473ms/step): Near-tie; marginal HBM contention difference.

3. **Training stability**: FSDP2 completed successfully; MFSDP+UB crashed on both runs due to NCCL
   window registration failure (`Failed to window register segment`). UB+manual-registration is not
   yet cluster-stable.

---

## 3. Improvement Opportunities

### For Automodel FSDP2

1. **Adopt contiguous flat parameter buffer** — Eliminates `chunk_cat` (476ms/step before each RS)
   and `split_with_sizes_copy_out` (402ms/step post-AG). Even though mostly overlapped today, the
   serial `chunk_cat` path is a ceiling on RS latency improvement.

2. **Evaluate NCCL user buffer mode** — If FSDP2 can pre-register parameter buffers with NCCL, it
   could achieve similar AG/RS improvements without the flat buffer requirement.

### For Megatron MFSDP

1. **Fix `use-nccl-ub` cluster stability** — Both UB runs crashed with `DistBackendError: Failed to
   window register segment`. Requires debug of `--fsdp-manual-registration` on this cluster.

2. **Fix `tss` GEMM via output tensor alignment** — Recovers ~150ms/step (~1.8% wall time). GEMM
   outputs into flat buffer offset views prevent TMA epilogue (`tst`). Fix: ensure output sub-views
   are 16-byte aligned with contiguous strides, or pre-allocate aligned staging buffers.

3. **Add NVTX layer markers** — MFSDP+UB profile has no module-level annotations, making per-layer
   scheduling analysis impossible.

---

## 4. Detailed Evidence

### 4.1 Collective Kernel Detail

| | MFSDP+UB | FSDP2 (iter 8) | Winner |
|---|---|---|---|
| AG calls/step | 326 | 320 | Similar |
| AG avg duration (µs) | **8,851** | 10,316 | **UB −14.2%** |
| AG total/step (ms) | **2,886** | 3,301 | **UB −12.5%** |
| RS calls/step | 164 | 162 | Tie |
| RS avg duration (µs) | **17,267** | 22,013 | **UB −21.6%** |
| RS total/step (ms) | **2,832** | 3,566 | **UB −20.6%** |
| Kernel name (AG) | `ncclDevKernel_AllGather_RING_LL` | Same | — |
| Kernel name (RS) | `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL` | Same | — |

UB mode benefits both AG and RS via user-buffer pointers that enable direct-placement into
pre-registered parameter buffers, eliminating staging copies at both the send and receive side.

### 4.2 Full Kernel Breakdown per Step

| Category | MFSDP+UB cnt/step | MFSDP+UB ms/step | FSDP2 cnt/step | FSDP2 ms/step | Winner |
|---|---|---|---|---|---|
| AllGather | 326 | 2,886 | 320 | 3,301 | **UB −12.5%** |
| ReduceScatter | 164 | 2,832 | 162 | 3,566 | **UB −20.6%** |
| GEMM total | 2,566 | 7,251 | 4,166 | 6,500 | FSDP2 −10.4% |
| Attention FWD (cuDNN) | 320 | 203 | 320 | 189 | FSDP2 −6.9% |
| Attention BWD (cuDNN) | 160 | 270 | 160 | 272 | Near-tie |
| Buffer copy (pre-RS + post-AG) | 161 | **9** | 482 | **878** | UB **97× less** |
| RMSNorm | 1,286 | 70 | 1,286 | 85 | FSDP2 (SM margin effect) |
| RoPE | 960 | 47 | 960 | 50 | Near-tie |
| **Step wall time** | | **8,448** | | **8,583** | **UB −2.0%** |

Comm vs Compute:

| | MFSDP+UB | FSDP2 |
|---|---|---|
| Comm (AG+RS) | 5,718ms | 6,867ms |
| Compute (GEMM+attn) | 7,724ms | 6,961ms |

### 4.3 GEMM Per-Shape Breakdown

> **Reading guide**: Compare **total ms/step**, not avg µs. MFSDP fuses QKV and gate+up, so call
> counts and FLOPs per call differ from FSDP2's separate projections. ³ marks non-apples-to-apples
> comparisons. ⁴ marks the `tss` vs `tst` epilogue difference on identical problem sizes.

| Kernel shape | MFSDP+UB ms/step | MFSDP+UB avg µs | FSDP2 ms/step | FSDP2 avg µs | Winner |
|---|---|---|---|---|---|
| `t_t_320x128_TNT` | 2,349 | 3,659 | 1,938 | 3,018 | Not apples-to-apples ³ |
| `t_t_256x128_NNT` | 1,352 | 2,804 | 1,318 | 1,370 | Near-tie |
| `t_t_256x128_TNT` | 1,263 | 1,974 | 944 | 738 | FSDP2 −25.3% |
| `tss_192x192_NTN` v1 | 1,129 | **3,507** | 973 (`tst`) | **3,022** | FSDP2 −13.8% (`tss` vs `tst`) ⁴ |
| `tss_192x192_NTN` v2 | 513 | **3,204** | 507 (`tst`) | **3,165** | Near-tie (`tss` vs `tst`) ⁴ |
| `t_t_320x128_NNT` | 507 | 3,172 | 503 | 3,142 | Tie |
| `tss_320x128_NTT` | 138 | 861 | 278 (`tst`) | 870 | FSDP2 more calls (separate KV) ³ |
| **Total GEMM** | **7,251** | | **6,500** | | **FSDP2 −10.4%** |

³ **`t_t_320x128_TNT` and `tss_320x128_NTT`: different FLOPs per call — not a fair comparison.**
- MFSDP fuses QKV into `[8192, 10240]` (one GEMM) vs FSDP2 separate Q/K/V tensors.
- MFSDP fuses KV gradient → fewer NTT calls (160/step) vs FSDP2 (320/step for separate K+V).

⁴ **`tss` vs `tst` on 192×192 FFN tiles — same problem size, different epilogue.**
GEMM outputs write into offset views of the flat contiguous parameter/gradient buffer. These sub-views
have non-standard base pointers → cuBLAS cannot construct TMA tile descriptors → falls back to `tss`
(shared-memory epilogue). NCCL UB mode does not alter parameter tensor layout. Total `tss` penalty:
~157ms/step (v1: 1,129ms vs 973ms = 156ms; v2: 513ms vs 507ms = 6ms; NTT: included in ³).
If `tss` were fixed, MFSDP GEMM would drop from 7,251ms → ~7,094ms — reducing the FSDP2 advantage
from −10.4% to −8.3%. Root cause fix: ensure flat-buffer sub-views are 16-byte aligned.

### 4.4 Attention Kernels

Both use identical cuDNN WGMMA kernels — fully fair comparison.

| | MFSDP+UB | FSDP2 (iter 8) |
|---|---|---|
| FWD kernel | `cudnn_...sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_4x` | Same |
| BWD kernel | `cudnn_...sdpa_sm90_flash_bprop_wgmma_f16_knob_26_64x64x128_1x` | Same |
| FWD calls/step | 320 | 320 |
| BWD calls/step | 160 | 160 |
| FWD avg_us / total/step | 635µs / 203ms | 591µs / 189ms |
| BWD avg_us / total/step | 1,686µs / 270ms | 1,701µs / 272ms |
| **Total attn/step** | **473ms** | **461ms** |

Near-tie on BWD. FWD is marginally faster in FSDP2 (591µs vs 635µs/call); likely from reduced HBM
contention when AG/RS buffers are smaller (individual tensors vs flat buffer).

### 4.5 Buffer Copy Overhead

| Kernel | Role | FSDP2 ms/step | MFSDP+UB ms/step |
|---|---|---|---|
| `chunk_cat_cuda_kernel` | Pre-RS: pack per-param grads into flat buffer | **476** | 0 |
| `split_with_sizes_copy_out` | Post-AG: unpack flat buffer back to param tensors | **402** | 0 |
| `CatArrayBatchedCopy_alignedK_contig` | MFSDP pre-AG shard copy from flat buffer | 0 | **9** |
| **Total** | | **878** | **9** |

FSDP2 iter 8 exact confirms prior estimates: `chunk_cat` 476ms + `split_with_sizes_copy_out` 402ms
= 878ms/step. MFSDP's flat buffer requires only a shard copy into the send buffer (9ms).

### 4.6 Comm–Compute Overlap Analysis

Source: `analyze_comm_compute_overlap.py`, 3-bucket decomposition.
MFSDP+UB: N_STEPS=2, full window. FSDP2: percentages from full-window script; absolute values
computed as percentage × iter 8 kernel time.

| Bucket | MFSDP+UB AG | FSDP2 AG | MFSDP+UB RS | FSDP2 RS |
|---|---|---|---|---|
| Comm merged wall/step | 2,886ms | 3,301ms | 2,832ms | 3,566ms |
| **A. Hidden by GEMM+attn** | **92.1%** | **88.5%** | **92.3%** | **86.6%** |
| **B. Hidden by norm / rope / elem** | 5.8% | 9.4% | 4.5% | 10.6% |
| **C. Truly exposed** | **2.1% (61ms)** | **2.0% (66ms)** | **3.1% (89ms)** | **2.7% (96ms)** |
| **Total hidden (A+B)** | 97.9% | 97.9% | 96.9% | 97.2% |

**Total truly exposed comm per step:**

| | MFSDP+UB | FSDP2 |
|---|---|---|
| AG exposed | **61ms** | 66ms |
| RS exposed | **89ms** | 96ms |
| **Total exposed** | **150ms** | **162ms** |
| **Winner** | **UB −7.4%** | |

**Key observations:**

1. **Both runs achieve ~97% comm-compute overlap.** Unlike MoE models (where FSDP2 AG is 98.3%
   *exposed*), dense LLaMA3-70B with full-recompute achieves excellent hiding in both frameworks.
   The very large GEMM windows (7,251ms UB, 6,500ms FSDP2) dwarf the comm duration, providing
   ample cover for all collectives.

2. **UB advantage primarily from kernel speed, not overlap.** The 12ms/step overlap advantage is
   minimal. The 1,149ms/step comm kernel time advantage (5,718 vs 6,867ms) is the primary driver.

3. **Bucket A (GEMM+attn) is dominant in both**: 88–92% of comm is buried inside GEMM+attn
   windows. The remaining exposed comm is small noise-level gaps between GEMM calls.

4. **Correction vs `analysis_report_ub.md`**: The prior report used FSDP2 overlap values from
   `analysis_report_no_flag.md` which had incorrect normalization (n=1.5 applied to already-wrong
   raw values). Correct per-step values for FSDP2 (iter 8 exact):
   - AG merged wall: **3,301ms** (was 4,473ms in prior report — off by 35%)
   - RS merged wall: **3,566ms** (was 3,861ms — off by 8%)
   - AG truly exposed: **66ms** (was 288ms — off by 4.4×)
   - RS truly exposed: **96ms** (was 121ms — off by 26%)

### 4.7 On `tss` vs `tst` GEMM Epilogue in UB Run

The `tss` epilogue persists on 192×192 NTN tiles, confirming the root cause is the output tensor
memory layout (flat buffer offset views), not any NCCL-related configuration. NCCL UB mode does not
affect GEMM kernel selection.

FSDP2 uses `tst` for the same 192×192 NTN problem size (322 calls, 973ms/step avg 3,022µs).
MFSDP `tss_192x192_NTN` v1: 322 calls, 1,129ms/step avg 3,507µs — 16% slower per call.

The structural fix remains: ensure flat-buffer sub-views exposed to cuBLAS have 16-byte-aligned base
pointers and contiguous strides, enabling TMA epilogue selection.

---

## 5. Estimated Contribution of UB Flags

Compared to the no_flag run (same config, no NCCL env vars, same cluster — not a controlled ablation).

### Estimated delta: MFSDP no_flag → MFSDP+UB

| Metric | MFSDP no_flag (est.) | MFSDP+UB | Δ |
|---|---|---|---|
| Step time (nsys) | ~8,931ms | 8,448ms | **−5.4% (−483ms)** |
| AllGather/step | ~4,130ms | 2,886ms | **−30.1% (−1,244ms)** |
| ReduceScatter/step | ~3,524ms | 2,832ms | **−19.6% (−692ms)** |
| Total comm/step | ~7,654ms | 5,718ms | **−25.3% (−1,936ms)** |
| Total GEMM/step | ~7,226ms | 7,251ms | essentially unchanged (+0.3%) |
| Truly exposed comm | ~582ms | 150ms | **−74% (−432ms)** |

**Comm improvement (~1,936ms/step)** driven by `--use-nccl-ub --fsdp-manual-registration`:
zero-copy parameter shards eliminate staging-copy round-trips at each AG and RS launch.

**GEMM is unchanged** confirming `--use-nccl-ub` and `--fsdp-double-buffer` have no side effects
on compute kernels.

**Overlap improvement (~432ms less exposed comm vs no_flag)**: shorter collectives more easily
buried inside GEMM windows; user-buffer mode allows collective start without pre-copy staging delay.

### Summary of flag value

| Flag | Primary effect | Estimated benefit |
|---|---|---|
| `--use-nccl-ub` | Zero-copy AG/RS via registered buffers | ~−1,900ms/step comm; −5.4% wall time |
| `--fsdp-manual-registration` | Pre-registers all parameter buffers at init | Required for UB mode; no independent effect |
| `--fsdp-double-buffer` | Double-buffers weights for AG prefetch | Contributes to overlap; not individually measured |

**Net estimated value of UB mode**: ~−5.4% wall time vs MFSDP without UB, and ~−2.0% vs FSDP2.
The primary risk is cluster stability: `--fsdp-manual-registration` crashes on this cluster.

---

## 6. Memory Analysis

| Metric | MFSDP+UB | FSDP2 |
|---|---|---|
| **Peak allocated** | **63.36 GB** | **78.57 GB** |
| H100 capacity (79.1 GB) | 80.1% | 99.3% |
| Headroom | ~15.7 GB | ~0.5 GB |

MFSDP+UB uses **15.2 GB less** than FSDP2. Root cause: FSDP2 temporarily holds both the original
per-parameter tensors and a packed flat send buffer live simultaneously during AG (`chunk_cat`
output), creating 2× peak shard footprint. MFSDP reads directly from the pre-allocated flat buffer;
no duplication occurs.

FSDP2's 99.3% utilization leaves virtually no headroom for longer sequences or larger MBS.

---

## 7. Stability Note

`--use-nccl-ub --fsdp-manual-registration` is **not cluster-stable** on this cluster. Both runs
failed with:
```
torch.distributed.DistBackendError: NCCL error in NCCLUtils.cpp:502
  Failed to window register segment with ptr 0x5984000000, size 33554432 on ncclComm_
```

The nsys profile was captured at steps 10–11, before the crash, so the performance data is valid.
The step time (8,448ms from nsys) should be treated as an optimistic estimate without log-based
confirmation.

Recommendation: debug `manual_buffer_registration` on this cluster before relying on UB results.
One path: try without `--fsdp-manual-registration` (let NCCL handle registration implicitly) and
verify if the `ncclAllGather` user-buffer path still engages.
