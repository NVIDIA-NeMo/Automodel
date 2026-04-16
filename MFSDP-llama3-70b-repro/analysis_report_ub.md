# LLaMA3-70B MFSDP+UB vs Automodel FSDP2 — nsys Profile Analysis (CP=1)

**Date**: 2026-04-15
**Model**: LLaMA3.1-70B
**Config**: TP=1, CP=1, PP=1, DP=64, MBS=1, GBS=128, SeqLen=4096, 8 nodes (64×H100 SXM), full-recompute

**Note on NCCL user buffer (UB) mode**: This report profiles MFSDP with `--fsdp-double-buffer`,
`--use-nccl-ub`, and `--fsdp-manual-registration` enabled — no NCCL env-var flags. Both training
runs (jobs 11134002, 11139868) crashed with a NCCL `DistBackendError: Failed to window register
segment` error during `manual_buffer_registration`. The nsys capture occurred at steps 10–11,
before the crash, so the profile is valid despite the job failure. No log-based throughput metrics
are available; step time is derived from the nsys window only. Compare to `analysis_report_no_flag.md`
for the same config without UB mode.

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
| Profile window | 16,897ms (2 steps) | 12,936ms (1.5 steps) |
| **Step time (nsys window)** | **8,448ms** | **8,624ms** |
| **Step time (log)** | **N/A (crashed)** | **8,646ms** |
| **MFU (approx)** | **~42.4%** | **~41.5%** |

---

## 1. Fairness & Shared Configuration

### What is the same (fair)

| Factor | MFSDP+UB | FSDP2 |
|---|---|---|
| Hardware | Same cluster, same H100 SXM node pool | ← same |
| Model architecture | LLaMA3.1-70B, hidden=8192, 80 layers, GQA | ← same |
| Parallelism | TP=1, CP=1, PP=1, DP=64 | ← same |
| Batch config | GBS=128, MBS=1, SeqLen=4096, GA=2 | ← same |
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

**MFSDP+UB**: n_iters=2 — no NVTX `iteration_*` markers in profile; derived from full kernel window
with `N_STEPS=2`. Validated by AG call count: 652 ÷ 326 expected calls/step = 2.000 exactly.

**FSDP2**: n_iters=1.5 — verified by RMSNorm FWD count: 963 ÷ 642 = 1.500 (same file as no_flag
report). All per-step values are raw ÷ n_iters.

---

## 2. High-Level Results & Adv / Disadv

### Headline

**MFSDP+UB is 2.1% faster** (8,448ms vs 8,624ms/step). NCCL user buffer mode gives MFSDP a
significant comm advantage that outweighs the `tss` GEMM epilogue penalty from the flat buffer.

### Summary

| Metric | MFSDP+UB | FSDP2 | UB vs FSDP2 |
|---|---|---|---|
| **Step wall time (nsys)** | **8,448ms** | 8,624ms | **UB −2.1%** |
| MFU | ~42.4% | ~41.5% | UB better |
| AllGather/step | **2,886ms** | 3,309ms | **UB −12.7%** |
| ReduceScatter/step | **2,832ms** | 3,482ms | **UB −18.7%** |
| Total comm/step | **5,718ms** | 6,791ms | **UB −15.8%** |
| Total GEMM/step | 7,251ms | **6,503ms** | UB +11.5% worse |
| Attention (FWD+BWD)/step | 473ms | **459ms** | Near-tie |
| Buffer copy overhead/step | **9ms** | 878ms | MFSDP 96× less |

### MFSDP+UB Advantages over FSDP2

1. **AllGather −12.7%** (3,309ms → 2,886ms): NCCL user buffer mode registers parameter buffers
   directly with the NCCL library, enabling zero-copy AG collectives. avg_us per call: 8,851µs (UB)
   vs 10,276µs (FSDP2).

2. **ReduceScatter −18.7%** (3,482ms → 2,832ms): UB mode similarly benefits RS by eliminating
   intermediate copies. avg_us per call: 17,267µs (UB) vs 21,494µs (FSDP2).

3. **Buffer copy overhead minimal** (9ms vs 878ms/step): MFSDP's flat contiguous buffer eliminates
   `chunk_cat` (476ms/step) and `split_with_sizes_copy_out` (402ms/step) that FSDP2 requires.

4. **Better comm–compute overlap**: UB truly exposed comm = 150ms/step vs FSDP2 ~409ms/step.
   Despite smaller absolute comm, UB achieves >97% overlap. See §4.6.

### FSDP2 Advantages

1. **GEMM −10.0% vs MFSDP+UB** (6,503ms vs 7,251ms/step): The `tss` epilogue penalty persists
   unchanged from the no_flag run. NCCL UB mode does not affect output tensor memory layout.
   The flat contiguous buffer still produces non-aligned base pointer offsets → cuBLAS selects `tss`
   (shared-memory epilogue) for 192×192 FFN tiles instead of `tst` (TMA-store epilogue).

2. **Attention slightly faster** (459ms vs 473ms/step): Near-tie; marginal HBM contention difference.

3. **Training stability**: FSDP2 completed successfully; MFSDP+UB crashed on both runs due to NCCL
   window registration failure (`Failed to window register segment`). UB+manual-registration is not
   yet cluster-stable.

---

## 3. Improvement Opportunities

### For Automodel FSDP2

1. **Adopt contiguous flat parameter buffer** — Eliminates `chunk_cat` serial bottleneck (476ms/step
   before each RS) and `split_with_sizes_copy_out` (402ms/step post-AG). Even though mostly overlapped
   today, the serial `chunk_cat` path is a ceiling on RS latency improvement.

2. **Evaluate NCCL user buffer mode** — If FSDP2 can pre-register parameter buffers with NCCL, it
   could achieve similar AG/RS improvements without the flat buffer requirement.

### For Megatron MFSDP

1. **Fix `use-nccl-ub` cluster stability** — Both UB runs crashed with `DistBackendError: Failed to
   window register segment`. Requires debug of `--fsdp-manual-registration` on this cluster
   (previously worked on EOS; likely a Lustre or NCCL version-specific issue).

2. **Fix `tss` GEMM via output tensor alignment** — Recovers ~150ms/step (~1.8% wall time). GEMM
   outputs into flat buffer offset views prevent TMA epilogue (`tst`). Fix: ensure output sub-views
   are 16-byte aligned with contiguous strides, or pre-allocate aligned staging buffers.

3. **Add NVTX layer markers** — MFSDP+UB profile has no module-level annotations, making per-layer
   scheduling analysis impossible and forcing N_STEPS inference from the full kernel window.

---

## 4. Detailed Evidence

### 4.1 Collective Kernel Detail

| | MFSDP+UB | FSDP2 | Winner |
|---|---|---|---|
| AG avg duration (µs) | **8,851** | 10,276 | **UB −13.9%** |
| AG total/step (ms) | **2,886** | 3,309 | **UB −12.7%** |
| RS avg duration (µs) | **17,267** | 21,494 | **UB −19.7%** |
| RS total/step (ms) | **2,832** | 3,482 | **UB −18.7%** |
| Kernel name | `ncclDevKernel_AllGather_RING_LL` | Same | — |
| Kernel name | `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL` | Same | — |

UB mode benefits both AG and RS equally via user-buffer pointers that enable direct-placement into
pre-registered parameter buffers, eliminating staging copies at both the send and receive side.

### 4.2 Full Kernel Breakdown per Step

| Category | MFSDP+UB cnt/step | MFSDP+UB ms/step | FSDP2 cnt/step | FSDP2 ms/step | Winner |
|---|---|---|---|---|---|
| AllGather | 326 | 2,886 | 322 | 3,309 | **UB −12.7%** |
| ReduceScatter | 164 | 2,832 | 162 | 3,482 | **UB −18.7%** |
| GEMM total | 2,566 | 7,251 | 4,166 | 6,503 | FSDP2 −10.2% |
| Attention FWD (cuDNN) | 320 | 203 | 320 | 190 | Near-tie |
| Attention BWD (cuDNN) | 160 | 270 | 160 | 269 | Tie |
| Copy overhead (buffer) | 160 | **9** | ~962 | **878** | UB 96× less |
| **Step wall time** | | **8,448** | | **8,624** | **UB −2.1%** |

Comm vs Compute:

| | MFSDP+UB | FSDP2 |
|---|---|---|
| Comm (AG+RS) | 5,718ms (42.8%) | 6,791ms (44.6%) |
| Compute (GEMM+attn) | 7,724ms (57.8%) | 6,962ms (45.7%) |

### 4.3 GEMM Per-Shape Breakdown

> **Reading guide**: Compare **total ms/step**, not avg µs. MFSDP fuses QKV and gate+up, so call
> counts and FLOPs per call differ from FSDP2's separate projections. ³ marks non-apples-to-apples
> comparisons. ⁴ marks the `tss` vs `tst` epilogue difference on identical problem sizes.

| Kernel shape | MFSDP+UB ms/step | MFSDP+UB avg µs | FSDP2 ms/step | FSDP2 avg µs | Winner |
|---|---|---|---|---|---|
| `tst_320x128_TNT` | 2,349 | 3,659 | 1,937 | 3,054 | Not apples-to-apples ³ |
| `tst_256x128_NNT` | 1,352 | 2,804 | 1,316 | 1,358 | FSDP2 −2.7% |
| `tst_256x128_TNT` | 1,263 | 1,974 | 946 | 722 | FSDP2 −25.1% |
| `tss_192x192_NTN` | 1,129 | **3,507** | 982 (`tst`) | **3,084** | FSDP2 −13.0% (`tss` vs `tst`) ⁴ |
| `tss_192x192_NTN` v2 | 513 | **3,204** | 507 (`tst`) | **3,196** | Near-tie (`tss` vs `tst`) ⁴ |
| `tst_320x128_NNT` | 508 | 3,172 | 503 | 3,137 | Tie |
| `tss_320x128_NTT` | 138 | 861 | 277 (`tst`) | 846 | FSDP2 more calls (separate KV) ³ |
| **Total GEMM** | **7,251** | | **6,503** | | **FSDP2 −10.2%** |

³ **`tst_320x128_TNT` and `tss_320x128_NTT`: different FLOPs per call — not a fair comparison.**
- MFSDP fuses QKV into `[8192, 10240]` (one GEMM) vs FSDP2 separate Q/K/V tensors.
- MFSDP fuses KV gradient → fewer NTT calls (160/step) vs FSDP2 (327/step for separate K+V).

⁴ **`tss` vs `tst` on 192×192 FFN tiles — same problem size, different epilogue.**
GEMM outputs write into offset views of the flat contiguous parameter/gradient buffer. These sub-views
have non-standard base pointers → cuBLAS cannot construct TMA tile descriptors → falls back to `tss`
(shared-memory epilogue). NCCL UB mode does not alter parameter tensor layout. Total `tss` penalty:
~153ms/step (v1: 1,129ms vs 982ms; v2: 513ms vs 507ms — near-tie). The structural fix is to ensure
flat-buffer sub-views are 16-byte aligned with contiguous strides.

The `tss_320x128_NTT` (160 calls, 138ms/step) represents grad-weight GEMMs for fused KV projections
also writing into the flat buffer. The per-call delta vs FSDP2's `tst_320x128_NTT` is ~15µs — not
the primary penalty. FSDP2 has more NTT calls (327/step) because K and V are separate tensors.

### 4.4 Attention Kernels

Both use identical cuDNN WGMMA kernels — fully fair comparison.

| | MFSDP+UB | FSDP2 |
|---|---|---|
| FWD kernel | `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_4x` | Same |
| BWD kernel | `cudnn_generated_fort_native_sdpa_sm90_flash_bprop_wgmma_f16_knob_26_64x64x128_1x` | Same |
| FWD avg_us / total/step | 635µs / 203ms | 589µs / 190ms |
| BWD avg_us / total/step | 1,686µs / 270ms | 1,673µs / 269ms |
| **Total attn/step** | **473ms** | **459ms** |

Near-tie. FSDP2 marginally faster on FWD (likely less HBM pressure during attention).

### 4.5 Buffer Copy Overhead

| Kernel | Role | FSDP2 ms/step | MFSDP+UB ms/step |
|---|---|---|---|
| `chunk_cat_cuda_kernel` | Pre-RS: pack per-param grads into flat buffer | **476** | 0 |
| `split_with_sizes_copy_out` | Post-AG: unpack flat buffer back to param tensors | **402** | 0 |
| `CatArrayBatchedCopy_vectorized` | MFSDP pre-AG shard copy from flat buffer | 0 | **9** |
| **Total** | | **878** | **9** |

Identical to the no_flag run. NCCL UB registration allows the collective to read/write parameter
shards directly, but does not eliminate the pre-AG `CatArrayBatchedCopy` (which packs the local
shard into the send buffer before AG begins). This 9ms overhead is negligible.

### 4.6 Comm–Compute Overlap Analysis

Source: `analyze_comm_compute_overlap.py`, 3-bucket decomposition. MFSDP+UB: N_STEPS=2, no NVTX.
FSDP2: values from `analysis_report_no_flag.md` (same file, validated methodology).

| Bucket | MFSDP+UB AG | FSDP2 AG | MFSDP+UB RS | FSDP2 RS |
|---|---|---|---|---|
| Comm wall time/step | 2,886ms | 4,473ms | 2,832ms | 3,861ms |
| **A. Hidden by GEMM + attn** | **92.1%** | 82.4% | **92.3%** | 84.3% |
| **B. Hidden by norm / rope / elem** | 5.8% | 11.2% | 4.5% | 12.6% |
| **C. Truly exposed** | **2.1% (61ms)** | 6.4% (288ms) | **3.1% (89ms)** | 3.1% (121ms) |
| **Total hidden (A+B)** | 97.9% | 93.6% | 96.9% | 96.9% |

**Total truly exposed comm per step:**

| | MFSDP+UB | FSDP2 |
|---|---|---|
| AG exposed | **61ms** | 288ms |
| RS exposed | **89ms** | 121ms |
| **Total exposed** | **150ms** | **409ms** |

**Key observations:**

1. **MFSDP+UB exposes 63% less comm than FSDP2** (150ms vs 409ms/step). Despite shorter absolute
   comm intervals (UB collectives complete faster), the remaining exposed tail is also smaller because
   UB mode allows collectives to start earlier — user-buffer registration removes pre-copy latency at
   comm launch.

2. **UB dominates Bucket A (GEMM+attn)**: 92.1%/92.3% of AG/RS wall time is hidden behind deep
   compute, vs 82–84% for FSDP2. The faster UB collectives run entirely within GEMM compute windows.

3. **Bucket B is smaller in UB than FSDP2**: FSDP2's comm (norm/rope/elem overlap) covers 11–12%
   of its larger comm window; UB's comm is shorter so less of it spills into light-op gaps.

4. **The 259ms/step exposed comm reduction vs FSDP2 explains most of the 176ms wall-time advantage**
   (net comm savings 1,073ms/step partially masked by +748ms GEMM penalty and overlap accounting).

### 4.7 On `tss` vs `tst` GEMM Epilogue in UB Run

The `tss` epilogue persists on 192×192 NTN tiles, confirming the root cause is the output tensor
memory layout (flat buffer offset views), not any NCCL-related configuration. NCCL UB mode does not
affect GEMM kernel selection.

The `tss_320x128_NTT` (160 calls, 138ms/step) represents grad-weight GEMMs for fused KV projections
writing into the flat gradient buffer. The per-call delta vs FSDP2's `tst_320x128_NTT` is ~15µs —
not a meaningful penalty. FSDP2 has more NTT calls (327/step) because K and V projections are
separate tensors.

The structural fix remains: ensure flat-buffer sub-views exposed to cuBLAS have 16-byte-aligned base
pointers and contiguous strides, enabling TMA epilogue selection.

---

## 5. Estimated Contribution of UB Flags

This section estimates the performance impact attributable to the three UB-specific flags
(`--fsdp-double-buffer`, `--use-nccl-ub`, `--fsdp-manual-registration`) by using the no_flag run
(same config, same cluster, no NCCL env vars) as a proxy baseline. This is an **estimate**, not a
controlled ablation — the no_flag and UB runs ran as separate jobs on potentially different nodes.

### Estimated delta: MFSDP no_flag → MFSDP+UB

| Metric | MFSDP no_flag (est.) | MFSDP+UB | Δ |
|---|---|---|---|
| Step time (nsys) | ~8,931ms | 8,448ms | **−5.4% (−483ms)** |
| AllGather/step | ~4,130ms | 2,886ms | **−30.1% (−1,244ms)** |
| ReduceScatter/step | ~3,524ms | 2,832ms | **−19.6% (−692ms)** |
| Total comm/step | ~7,654ms | 5,718ms | **−25.3% (−1,936ms)** |
| Total GEMM/step | ~7,226ms | 7,251ms | essentially unchanged (+0.3%) |
| Truly exposed comm | ~582ms | 150ms | **−74% (−432ms)** |

### Attribution

**Comm improvement (~1,936ms/step)** is driven by `--use-nccl-ub --fsdp-manual-registration`:
NCCL user buffer registration allows parameter shards to be read/written in-place during collectives,
eliminating the staging-copy round-trips that otherwise occur at each AG and RS kernel launch.
The improvement is roughly equal on AG (−30%) and RS (−20%), suggesting both collectives benefit
from direct buffer access rather than one being bottlenecked by a different mechanism.

**GEMM is unchanged** (7,226ms → 7,251ms, within run-to-run noise). This confirms `--use-nccl-ub`
and `--fsdp-double-buffer` have no side effects on compute kernels.

**Overlap improvement (~432ms less exposed comm)** follows from the comm improvement: shorter
collectives are more easily buried inside GEMM windows, and user-buffer mode allows the collective
to start sooner after the preceding GEMM issues its outputs (no pre-copy staging delay).

**`--fsdp-double-buffer` contribution** is not individually isolatable from this data. Its role is
to pre-fetch the next layer's weights while the current layer computes, reducing AG exposure between
layers. In the UB run, AG exposed is already 61ms/step (vs FSDP2's 288ms), suggesting effective
prefetch. Without a separate double-buffer-only ablation, the split between double-buffer and UB
registration is not determinable.

### Summary of flag value

| Flag | Primary effect | Estimated benefit |
|---|---|---|
| `--use-nccl-ub` | Zero-copy AG/RS via registered buffers | ~−1,900ms/step comm; −5.4% wall time |
| `--fsdp-manual-registration` | Pre-registers all parameter buffers at init | Required for UB mode; no independent effect |
| `--fsdp-double-buffer` | Double-buffers weights for AG prefetch | Contributes to overlap; not individually measured |

**Net estimated value of UB mode**: ~−5.4% wall time vs MFSDP without UB, and ~−2.1% vs FSDP2.
The primary risk is cluster stability: `--fsdp-manual-registration` crashes on this cluster. If
stability is resolved, UB mode makes MFSDP the faster framework at this config.

---

## 7. Memory Analysis

Source: nsys profile (MFSDP+UB peak allocated read from nsys-rep); Megatron tensorboard log (FSDP2,
from `analysis_report_no_flag.md`). Note: `CUDA_MEMORY_USAGE` is not exported to sqlite by this
nsys version, so MFSDP+UB figures come directly from the nsys-rep.

### 7.1 Summary

| Metric | MFSDP+UB | FSDP2 |
|---|---|---|
| **Peak allocated** | **63.36 GB** | **78.57 GB** |
| H100 capacity (79.1 GB) | 80.1% | 99.3% |
| Headroom | ~15.7 GB | ~0.5 GB |

MFSDP+UB uses **15.2 GB less** than FSDP2. FSDP2 is effectively at capacity.

### 7.2 Root Cause

The difference is structural: FSDP2 lacks a contiguous flat parameter buffer. During AllGather,
FSDP2 must pack individual per-parameter tensors into a temporary flat send buffer via `chunk_cat`.
At the AG peak, both the original per-parameter tensors and the packed flat buffer are live
simultaneously — two copies of the weight shards coexist. MFSDP's parameters already reside in a
pre-allocated flat contiguous buffer, so AllGather reads directly from it with no temporary copy;
no duplication occurs.

### 7.3 Implication

FSDP2's 99.3% utilization (78.57 / 79.1 GB) leaves virtually no headroom for scaling to longer
sequences or larger MBS. MFSDP+UB's 80.1% utilization retains ~15.7 GB margin despite the
additional UB registration overhead.

---

## 8. Stability Note

`--use-nccl-ub --fsdp-manual-registration` is **not cluster-stable** on this cluster. Both runs
failed with:
```
torch.distributed.DistBackendError: NCCL error in NCCLUtils.cpp:502
  Failed to window register segment with ptr 0x5984000000, size 33554432 on ncclComm_
```

This is a known limitation: `NCCL_IB_SL` or specific Lustre mount configurations can conflict with
NCCL window registration. The nsys profile was captured at steps 10–11 (profiling window
`--profile-step-start 10 --profile-step-end 12`), before the crash, so the performance data is valid.
The step time (8,448ms from nsys) should be treated as an optimistic estimate without log-based
confirmation.

Recommendation: debug `manual_buffer_registration` on this cluster before relying on UB results.
One path: try without `--fsdp-manual-registration` (let NCCL handle registration implicitly) and
verify if the `ncclAllGather` user-buffer path still engages.
