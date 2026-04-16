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

**MFSDP+UB is 2.1% faster** (8,448ms vs 8,624ms/step). UB mode closes the gap from the no_flag
run's 3.5% deficit against FSDP2, flipping the result. The communication improvement from NCCL user
buffer (−30% AG, −20% RS vs no_flag) exceeds the GEMM penalty that remains from the `tss` epilogue.

### Three-way summary

| Metric | MFSDP no_flag | MFSDP+UB | FSDP2 | UB vs FSDP2 |
|---|---|---|---|---|
| **Step wall time (nsys)** | 8,931ms | **8,448ms** | 8,624ms | **UB −2.1%** |
| MFU | ~40.1% | ~42.4% | ~41.5% | UB best |
| AllGather/step | 4,130ms | **2,886ms** | 3,309ms | **UB −12.7%** |
| ReduceScatter/step | 3,524ms | **2,832ms** | 3,482ms | **UB −18.7%** |
| Total comm/step | 7,654ms | **5,718ms** | 6,791ms | **UB −15.8%** |
| Total GEMM/step | 7,226ms | 7,251ms | **6,503ms** | UB +11.5% worse |
| Attention (FWD+BWD)/step | 482ms | 473ms | **459ms** | Near-tie |
| Buffer copy overhead/step | **9ms** | **9ms** | 878ms | MFSDP 96× less |

### MFSDP+UB Advantages over FSDP2

1. **AllGather −12.7%** (3,309ms → 2,886ms): NCCL user buffer mode registers parameter buffers
   directly with the NCCL library, enabling zero-copy AG collectives. Compared to no_flag MFSDP this
   is a −30.1% AG improvement (4,130ms → 2,886ms). avg_us per call: 12,669µs (no_flag) → 8,851µs (UB).

2. **ReduceScatter −18.7%** (3,482ms → 2,832ms): UB mode similarly benefits RS by eliminating
   intermediate copies. vs no_flag: −19.6% (3,524ms → 2,832ms). avg_us: 21,491µs (no_flag) → 17,267µs (UB).

3. **Buffer copy overhead parity with no_flag** (9ms): UB mode does not change MFSDP's flat
   contiguous buffer; `CatArrayBatchedCopy` overhead remains 9ms/step vs FSDP2's 878ms/step.

4. **Better comm–compute overlap**: UB truly exposed comm = 150ms/step vs FSDP2 ~409ms/step (from
   no_flag analysis; same FSDP2 file). Despite smaller absolute comm, UB achieves >97% overlap. See §4.8.

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

### 4.1 UB Mode Effect on Collectives — Controlled Experiment

| | MFSDP no_flag | MFSDP+UB | FSDP2 |
|---|---|---|---|
| AG avg duration (µs) | 12,669 | **8,851** | 10,276 |
| AG total/step (ms) | 4,130 | **2,886** | 3,309 |
| AG vs FSDP2 | MFSDP 24.9% slower | **UB 12.7% faster** | — |
| RS avg duration (µs) | 21,491 | **17,267** | 21,494 |
| RS total/step (ms) | 3,524 | **2,832** | 3,482 |
| RS vs FSDP2 | MFSDP 1.2% slower | **UB 18.7% faster** | — |

UB mode improves both AG and RS, unlike `NCCL_P2P_NET_CHUNKSIZE`+`NCCL_IB_SL` flags (which mainly
reduced RS). Both use the same `ncclDevKernel_AllGather_RING_LL` and
`ncclDevKernel_ReduceScatter_Sum_f32_RING_LL` kernel names, but with user-buffer pointers that
enable direct-placement without staging copies.

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
The flat buffer root cause is identical to the no_flag run: GEMM output sub-views into the flat
buffer have non-standard base pointers → cuBLAS cannot construct TMA tile descriptors → falls back
to `tss`. NCCL UB mode does not alter parameter tensor layout. Total `tss` penalty: ~153ms/step
(difference between MFSDP+UB `tss_192x192_NTN` total and FSDP2 `tst_192x192_NTN` total, adjusted
for avg_us). Note: v2 (512.7ms vs 506.7ms) is near-tie; v1 (1,129ms vs 982ms) drives the penalty.

**Comparison of MFSDP+UB vs no_flag GEMM**:
- GEMM total: 7,251ms (UB) vs 7,226ms (no_flag) — essentially identical (+0.3%)
- NCCL UB mode has no effect on GEMM performance, as expected.
- The `tss_320x128_NTT` kernel (138ms/step) appears in the UB run; it was present in the no_flag
  run as well but not highlighted in that report (it corresponds to small grad-weight GEMMs for
  fused KV projections that also write into the flat buffer).

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

| | MFSDP no_flag | MFSDP+UB | FSDP2 |
|---|---|---|---|
| AG exposed | 334ms | **61ms** | 288ms |
| RS exposed | 248ms | **89ms** | 121ms |
| **Total exposed** | **582ms** | **150ms** | **409ms** |

**Key observations:**

1. **MFSDP+UB has the least exposed comm of all three configurations** (150ms/step vs 409ms
   FSDP2, 582ms no_flag). Despite smaller comm totals (UB collectives complete faster), the
   remaining exposed tail is also smaller because UB mode allows collectives to start earlier
   (user-buffer registration removes pre-copy latency at comm launch).

2. **UB dominates Bucket A (GEMM+attn)**: 92.1%/92.3% of AG/RS wall time is hidden behind deep
   compute, vs 82–84% for FSDP2 and 84–90% for no_flag MFSDP. The faster UB collectives run
   entirely within GEMM compute windows.

3. **Bucket B is smaller in UB than FSDP2**: FSDP2's lighter comm (norm/rope/elem overlap) covers
   11–12% of its larger comm window; UB's comm is shorter so less of it spills into light-op gaps.

4. **The 259ms/step exposed comm reduction vs FSDP2 (409ms → 150ms) contributes to most of the
   176ms wall-time advantage** (along with the net comm savings: 6,791ms → 5,718ms = 1,073ms comm
   reduction, but partially masked by the +748ms GEMM penalty and overlap effects).

### 4.7 On `tss` vs `tst` GEMM Epilogue in UB Run

The `tss` epilogue persists on 192×192 NTN tiles in the UB run, confirming the root cause is the
output tensor memory layout (flat buffer offset views), not any NCCL-related configuration.

A new `tss_320x128_NTT` appears explicitly in the UB profile (160 calls, 138ms/step). This kernel
was likely also present in the no_flag run but fell below the reporting threshold. It represents
grad-weight computations for fused KV projections writing into the flat gradient buffer. The per-call
delta vs FSDP2's `tst_320x128_NTT` is small (~15µs/call), so this is not the primary penalty.

The structural fix remains: ensure flat-buffer sub-views exposed to cuBLAS have 16-byte-aligned base
pointers and contiguous strides, enabling TMA epilogue selection.

---

## 5. Memory Analysis

Memory analysis for MFSDP+UB is not available — both training runs crashed before logging steady-state
memory metrics via `--log-memory-to-tensorboard`. The peak allocation during `manual_buffer_registration`
may additionally exceed the no_flag run's 46.2 GB/GPU if NCCL pre-registers large staging buffers.

For reference from `analysis_report_no_flag.md`:

| Metric | MFSDP no_flag | FSDP2 |
|---|---|---|
| **Peak allocated** | **46.2 GB** | **53.1 GB** |
| **Reserved** | **61.6 GB** | **68.1 GB** |
| H100 capacity utilization | 77% | 85% |

MFSDP+UB memory footprint is expected to be ≥ no_flag (UB registration may add per-buffer metadata),
but likely still below FSDP2's 53.1 GB since the flat buffer structure is unchanged.

---

## 6. Stability Note

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
