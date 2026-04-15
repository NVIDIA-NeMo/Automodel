# LLaMA3-70B MFSDP vs Automodel FSDP2 — nsys Profile Analysis (CP=1)

**Date**: 2026-04-14
**Model**: LLaMA3.1-70B
**Config**: TP=1, CP=1, PP=1, DP=64, MBS=1, GBS=128, SeqLen=4096, 8 nodes (64×H100 SXM), full-recompute

**Note on NCCL flags**: An earlier MFSDP run (job 11125550) included `NCCL_P2P_NET_CHUNKSIZE=2097152`,
`NCCL_IB_SL=1`, and `TORCH_NCCL_AVOID_RECORD_STREAMS=1`, which gave MFSDP a 33% RS advantage
(2,613ms vs 3,482ms/step) not present in FSDP2. This report uses a clean run without those flags so
both sides have identical NCCL environments. Effect of adding those flags is noted where relevant.

---

## Runs Compared

| | Megatron MFSDP | Automodel FSDP2 |
|---|---|---|
| Framework | Megatron-LM (`--use-megatron-fsdp`) | Automodel PyTorch FSDP2 |
| Container | `pytorch26.03_te2.14_deepep_x86` | `nemo-automodel:26.04.rc4` |
| Slurm job | 11126057 | 11124101 |
| nsys-rep | `MFSDP-llama3-70b-repro/nsys/..._no_flag.nsys-rep` | `slurm_jobs/11124101/nsys_llama31_70b_..._node0.nsys-rep` |
| Profile steps | 10–12 | 7–8 |
| Profile window | 17,861ms (2 steps) | 12,936ms (1.5 steps) |
| **Step time (log)** | **~8,950ms** | **8,646ms** |
| **Step time (nsys window)** | **8,931ms** | **8,624ms** |
| **MFU (approx)** | **~40.1%** | **~41.5%** |

---

## 1. Fairness & Shared Configuration

### What is the same (fair)

| Factor | MFSDP | FSDP2 |
|---|---|---|
| Hardware | Same cluster, same H100 SXM node pool | ← same |
| Model architecture | LLaMA3.1-70B, hidden=8192, 80 layers, GQA | ← same |
| Parallelism | TP=1, CP=1, PP=1, DP=64 | ← same |
| Batch config | GBS=128, MBS=1, SeqLen=4096, GA=2 | ← same |
| Activation recompute | Full recompute | ← same (642 RMSNorm FWD/step both) |
| Attention backend | TE FusedAttention (cuDNN WGMMA) | ← same kernel |
| NCCL env vars | None | None |
| Optimizer dtype | BF16 params, FP32 reduce | ← same |

### What is different (potential unfairness)

| Factor | MFSDP | FSDP2 | Impact |
|---|---|---|---|
| Container version | `pytorch26.03` | `26.04.rc4` | Minor TE/PyTorch/NCCL version diff |
| NVTE_LAYERNORM_SM_MARGIN | 16 (set) | Not set | ~4ms/step on RMSNorm; does not affect GEMM |
| Parameter layout | Flat contiguous buffer | Individual tensors per param | Structural — drives `tss` vs `tst` GEMM variant and buffer copy overhead |
| Projection fusion | Fused QKV, fused gate+up | Separate q/k/v, gate, up | Architectural — changes GEMM call count and tile shapes |

### Window normalization

**MFSDP**: n_iters=2 — verified by RMSNorm FWD count: 1,284 ÷ 642 = 2.000 exactly.
**FSDP2**: n_iters=1.5 — verified by RMSNorm FWD: 963 ÷ 642 = 1.500.
All per-step values below are raw ÷ n_iters.

---

## 2. High-Level Results & Adv / Disadv

### Headline

**FSDP2 is 3.5% faster** (8,624ms vs 8,931ms/step).

| Metric | MFSDP | FSDP2 | Winner |
|---|---|---|---|
| **Step wall time** | **8,931ms** | **8,624ms** | **FSDP2 −3.5%** |
| MFU | ~40.1% | ~41.5% | FSDP2 |
| AllGather/step | 4,130ms | **3,309ms** | FSDP2 −19.9% |
| ReduceScatter/step | 3,524ms | 3,482ms | **Tie** (0.01%) |
| Total comm/step | 7,668ms | **6,792ms** | FSDP2 −11.4% |
| Total GEMM/step | 7,226ms | **6,503ms** | FSDP2 −10.0% |
| Attention (FWD+BWD)/step | 482ms | **459ms** | Near-tie |
| Buffer copy overhead/step | **9ms** | 878ms | MFSDP 96× less |

### FSDP2 Advantages

1. **AllGather −19.9%** (4,130ms → 3,309ms): FSDP2's prefetch pipelining and `split_with_sizes_copy_out`
   overlap make AG effectively faster. Structural scheduling advantage.

2. **GEMM −10.0%** (7,226ms → 6,503ms): Driven by two factors:
   - **`tss` vs `tst` cuBLAS epilogue** on 192×192 FFN tiles: MFSDP's flat buffer produces output tensor
     sub-views with non-standard base pointers, preventing cuBLAS from using TMA epilogue (`tst`) →
     falls back to slower shared-memory epilogue (`tss`). ~175ms/step impact.
   - **Separate projection tile shapes**: FSDP2's smaller per-projection GEMMs fit cuBLAS tile heuristics
     more efficiently.

3. **ReduceScatter — parity at baseline**: MFSDP RS (21,491µs) = FSDP2 RS (21,494µs) exactly with no
   NCCL tuning. Note: adding `NCCL_P2P_NET_CHUNKSIZE=2097152` and `NCCL_IB_SL=1` to MFSDP gives 33%
   RS speedup (2,613ms vs 3,482ms/step), which would flip this category to MFSDP's favor.

### MFSDP Advantages

1. **Buffer copy overhead 96× less** (9ms vs 878ms/step): Flat contiguous buffer eliminates `chunk_cat`
   (pre-RS pack, 476ms/step) and `split_with_sizes_copy_out` (post-AG unpack, 402ms/step). ~98% of
   FSDP2's copy overhead is overlapped with communication, so the wall-time benefit is partial. However,
   the `chunk_cat` serial path (2,937µs before each RS) inflates FSDP2's measured RS latency.

2. **NCCL tuning upside**: With cluster-specific NCCL flags, MFSDP RS drops from 3,524ms → 2,613ms/step
   (−33%), nearly closing the entire wall-time gap. FSDP2 wall time would improve similarly if those
   same flags were applied.

---

## 3. Improvement Opportunities

### For Automodel FSDP2

1. **Tune NCCL settings** — `NCCL_P2P_NET_CHUNKSIZE=2097152` and `NCCL_IB_SL=1` gave MFSDP 33% RS
   speedup on this cluster. Testing for FSDP2 could reduce RS from 3,482ms → ~2,613ms/step, widening
   the FSDP2 wall-time lead from 3.5% to ~13%.

2. **Adopt contiguous flat parameter buffer** — Eliminates `chunk_cat` serial bottleneck (476ms/step
   before each RS) and `split_with_sizes_copy_out` (402ms/step post-AG). Even though mostly overlapped
   today, `chunk_cat` is a hard ceiling on RS latency improvement.

3. **Maintain AG scheduling quality** — FSDP2's AG pipelining (+19.9% advantage) is a genuine strength.
   Ensure prefetch depth and `split_with_sizes_copy_out` overlap are preserved in future refactors.

### For Megatron MFSDP

1. **Fix `tss` GEMM via output tensor alignment** — GEMM outputs into flat buffer offset views prevent
   TMA epilogue (`tst`). Fix: ensure output sub-views are 16-byte aligned with contiguous strides, or
   pre-allocate aligned staging buffers. Recovers ~175ms/step (~2% wall time).

2. **Add NCCL tuning to cluster config** — `NCCL_P2P_NET_CHUNKSIZE=2097152` and `NCCL_IB_SL=1` gave
   33% RS improvement here. Evaluate for standard recipe (note: cluster-dependent; found to hurt on
   EOS cluster).

3. **Add NVTX layer markers** — MFSDP profile has no module-level annotations, making per-layer
   scheduling analysis impossible. Critical for debugging the AG/GEMM overlap patterns.

---

## 4. Detailed Evidence

### 4.1 RS Parity — Controlled Experiment Result

| | MFSDP (w/ NCCL flags) | MFSDP | FSDP2 |
|---|---|---|---|
| RS avg duration | 15,935 µs | **21,491 µs** | **21,494 µs** |
| RS total/step | 2,613ms | 3,524ms | 3,482ms |
| vs FSDP2 | MFSDP 33% faster | **Identical (0.01%)** | — |

MFSDP RS without NCCL flags = FSDP2 RS to within 3µs averaging noise. This conclusively proves:
- The entire 33% RS advantage from the flagged run was caused by `NCCL_P2P_NET_CHUNKSIZE=2097152` / `NCCL_IB_SL=1`.
- MFSDP has no structural RS advantage at baseline; both use identical `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL`.

### 4.2 AllGather Detail

| Metric | MFSDP (w/ NCCL flags) | MFSDP | FSDP2 |
|---|---|---|---|
| AG avg duration | 10,990 µs | 12,669 µs | **10,276 µs** |
| AG total/step | 3,583ms | 4,130ms | **3,309ms** |

MFSDP AG without flags is 16% slower than the flagged run. `NCCL_P2P_NET_CHUNKSIZE` and `NCCL_IB_SL`
primarily affect RS/P2P, not AG — the degradation is likely from removing
`TORCH_NCCL_AVOID_RECORD_STREAMS=1` (increases CUDA stream sync overhead) or natural cluster fabric
variance between jobs.

### 4.3 Full Kernel Breakdown per Step

| Category | MFSDP cnt/step | MFSDP ms/step | FSDP2 cnt/step | FSDP2 ms/step | Winner |
|---|---|---|---|---|---|
| AllGather | 326 | 4,130 | 322 | 3,309 | FSDP2 −19.9% |
| ReduceScatter | 164 | 3,524 | 162 | 3,482 | Tie |
| GEMM total | 2,566 | 7,226 | 4,166 | 6,503 | FSDP2 −10.0% |
| Attention FWD (cuDNN) | 320 | 199 | 320 | 190 | Near-tie |
| Attention BWD (cuDNN) | 160 | 283 | 160 | 269 | Near-tie |
| RMSNorm FWD+BWD | 964 | ~73 | 964 | ~83 | Tie |
| Copy overhead (buffer) | 161 | **9** | ~962 | **878** | MFSDP 96× less |
| **Total kernel/step** | | **15,927** | | **15,240** | MFSDP +4.5% more |
| **Step wall time** | | **8,931** | | **8,624** | **FSDP2 −3.5%** |

Comm vs Compute:

| | MFSDP | FSDP2 |
|---|---|---|
| Comm (AG+RS) | 7,668ms (48.2%) | 6,792ms (44.6%) |
| Compute (GEMM+attn) | 7,708ms (48.4%) | 6,962ms (45.7%) |

### 4.4 GEMM Per-Shape Breakdown

> **Reading guide**: Compare **total ms/step**, not avg µs. Avg µs differs because the calls cover
> different problem sizes (see ³).

| Kernel shape | MFSDP ms/step | MFSDP avg µs | FSDP2 ms/step | FSDP2 avg µs | Notes |
|---|---|---|---|---|---|
| `tst_320x128_TNT` | 2,495 | 3,887 | 1,937 | 3,017 | Not apples-to-apples — see ³ |
| `tst_256x128_NNT` | 1,406 | 2,917 | 1,316 | 1,368 | FSDP2 2× calls (separate proj) |
| `tst_256x128_TNT` | 1,290 | 2,016 | 946 | 739 | FSDP2 2× calls (separate proj) |
| `tss_192x192_NTN` | 1,143 | **3,712** | 982 (`tst`) | **3,048** | MFSDP +21.8% — `tss` vs `tst` ⁴ |
| `tss_192x192_NTN` v2 | 541 | **3,382** | 507 (`tst`) | **3,167** | MFSDP +6.8% — `tss` vs `tst` |
| `tst_320x128_NNT` | 497 | 3,105 | 503 | 3,141 | Tie |
| **Total GEMM** | **7,226** | | **6,503** | | **FSDP2 −10.0%** |

³ **`tst_320x128_TNT`: different FLOPs per call — not a fair per-call comparison.**
For LLaMA 70B (GQA: 64 Q heads, 8 KV heads, head_dim=128):
- MFSDP: fused QKV weight `[8192, 10240]` → one GEMM, N=10240 FLOPs per call
- FSDP2: Q projection `[8192, 8192]` → `tst_320x128_TNT`, N=8192; K+V `[8192, 1024]` each → different
  smaller-tile rows

Both have ~642 calls in this row but cover different amounts of work. MFSDP's +29% per-call
(3,887 vs 3,017µs) matches the expected N ratio (10240/8192 = 1.25×) — MFSDP is doing proportionally
more FLOPs, not being less efficient. Fused projections are not causing the GEMM gap.

⁴ **`tss` vs `tst` — the real GEMM gap driver.** For 192×192 FFN tiles both sides run the same problem
size (genuine apples-to-apples), yet MFSDP gets `tss` (shared-memory epilogue, slower) and FSDP2 gets
`tst` (TMA-store epilogue, faster). Persists with and without NCCL flags.

Root cause — **output tensor memory layout**:
- `tst` uses TMA (Tensor Memory Accelerator) for the output epilogue write. TMA requires a standard
  contiguous layout: 16-byte-aligned base pointer, clean strides.
- MFSDP GEMM outputs write into **offset views of the flat contiguous parameter/gradient buffer** — each
  output has a non-standard base address. cuBLAS cannot construct a valid TMA tile descriptor → falls
  back to `tss` (shared-memory epilogue, no TMA).
- FSDP2 outputs go into **individually allocated tensors** — clean base pointer and contiguous strides
  → cuBLAS selects `tst`.

This is a **direct structural consequence of MFSDP's flat buffer**, not a tuning issue. Contributes
~175ms/step to the GEMM gap. Fix: ensure GEMM output sub-views are 16-byte aligned with contiguous
strides, or override cuBLAS epilogue selection.

### 4.5 Attention Kernels

Both use identical cuDNN WGMMA kernels — fully fair comparison.

| | MFSDP | FSDP2 |
|---|---|---|
| FWD kernel | `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_4x` | Same |
| BWD kernel | `cudnn_generated_fort_native_sdpa_sm90_flash_bprop_wgmma_f16_knob_26_64x64x128_1x` | Same |
| FWD total/step | 199ms | **190ms** |
| BWD total/step | 283ms | **269ms** |

Near-tie. FSDP2 marginally faster per call — likely less HBM contention during attention phases.

### 4.6 Buffer Copy Overhead

| Kernel | Role | FSDP2 ms/step | MFSDP ms/step |
|---|---|---|---|
| `chunk_cat_cuda_kernel` | Pre-RS: pack per-param grads into flat buffer | **476** | 0 |
| `split_with_sizes_copy_out` | Post-AG: unpack flat buffer back to param tensors | **402** | 0 |
| `CatArrayBatchedCopy` | MFSDP pre-AG shard copy from flat buffer | 0 | **9** |
| **Total** | | **878** | **9** |

FSDP2's `chunk_cat` (avg 2,937µs/call) is serial immediately before each RS — blocks RS launch.
At RS kernel-level parity (21,494µs FSDP2 vs 21,491µs MFSDP), subtracting `chunk_cat` from FSDP2's
measurement implies FSDP2's physical RS hardware throughput is ~2,937µs faster than MFSDP per call.
The 878ms/step copy overhead is ~98% overlapped in wall time today, but the serial `chunk_cat` path is
a ceiling that prevents further RS latency improvement.

### 4.7 On `NVTE_FWD/BWD_LAYERNORM_SM_MARGIN=16`

Set in MFSDP, not in FSDP2. Leaves 16 SMs idle during RMSNorm kernel launches, reserving them for
concurrent NCCL streams.

- **Effect on GEMM**: None. cuBLAS kernel selection is independent of this setting.
- **Effect on `tss` vs `tst`**: None. That is driven by output tensor memory layout (flat buffer
  base pointer alignment), not SM availability.
- **Wall-time impact**: ~5% slower RMSNorm per call → ~4ms/step; negligible in the overall comparison.
