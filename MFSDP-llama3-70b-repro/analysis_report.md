# LLaMA3-70B MFSDP (w/ NCCL flags) vs Automodel FSDP2 — nsys Profile Analysis (CP=1)

**Date**: 2026-04-15
**Model**: LLaMA3.1-70B
**Config**: TP=1, CP=1, PP=1, DP=64, MBS=1, GBS=128, SeqLen=4096, 8 nodes (64×H100 SXM), full-recompute

**Note on NCCL flags**: This MFSDP run includes `NCCL_P2P_NET_CHUNKSIZE=2097152`, `NCCL_IB_SL=1`, and
`TORCH_NCCL_AVOID_RECORD_STREAMS=1` which are absent in FSDP2. These flags give MFSDP a 33% RS
advantage (2,613ms vs 3,482ms/step) not present at baseline. See `analysis_report_no_flag.md` for the
fair no-flags comparison.

---

## Runs Compared

| | Megatron MFSDP (w/ NCCL flags) | Automodel FSDP2 |
|---|---|---|
| Framework | Megatron-LM (`--use-megatron-fsdp`) | Automodel PyTorch FSDP2 |
| Container | `pytorch26.03_te2.14_deepep_x86` | `nemo-automodel:26.04.rc4` |
| Slurm job | 11125550 | 11124101 |
| nsys-rep | `MFSDP-llama3-70b-repro/nsys/llama3_70b_mfsdp_tp1_pp1_ep1_cp1_hsdp1_alltoall_mbs1_gbs128_seqlen4096_cw_n8_full-recompute.nsys-rep` | `slurm_jobs/11124101/nsys_llama31_70b_pretrain_tp1cp1pp1_te_attn_cp1_no_compile_gbs128_seqlen4096_node0.nsys-rep` |
| Profile steps | 10–12 | 7–8 |
| Profile window | 17,834ms (2 steps) | 12,936ms (1.5 steps) |
| **Step time (log)** | **~8,910ms** | **8,646ms** |
| **Step time (nsys window)** | **8,917ms** | **8,624ms** |
| **MFU (approx)** | **~40.2%** | **~41.5%** |

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
| Optimizer dtype | BF16 params, FP32 reduce | ← same |

### What is different (potential unfairness)

| Factor | MFSDP | FSDP2 | Impact |
|---|---|---|---|
| **NCCL_P2P_NET_CHUNKSIZE** | **2097152 (set)** | **Not set** | **MFSDP RS 33% faster — primary unfairness** |
| **NCCL_IB_SL** | **1 (set)** | **Not set** | **IB service level — affects RS/P2P routing** |
| **TORCH_NCCL_AVOID_RECORD_STREAMS** | **1 (set)** | **Not set** | **Reduces CUDA stream sync overhead** |
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

**FSDP2 is 3.3% faster** (8,624ms vs 8,917ms/step) despite MFSDP's tuned NCCL RS advantage.

| Metric | MFSDP | FSDP2 | Winner |
|---|---|---|---|
| **Step wall time** | **8,917ms** | **8,624ms** | **FSDP2 −3.3%** |
| MFU | ~40.2% | ~41.5% | FSDP2 |
| AllGather/step | 3,583ms | **3,309ms** | FSDP2 −7.7% |
| ReduceScatter/step | **2,613ms** | 3,482ms | **MFSDP −25%** |
| Total comm/step | **6,210ms** | 6,792ms | MFSDP −8.5% |
| Total GEMM/step | 7,666ms | **6,503ms** | FSDP2 −15.2% |
| Attention (FWD+BWD)/step | 498ms | **459ms** | Near-tie |
| Buffer copy overhead/step | **9ms** | 878ms | MFSDP 96× less |

### FSDP2 Advantages

1. **GEMM −15.2%** (7,666ms → 6,503ms): Largest single gap. Driven by two factors:
   - **`tss` vs `tst` cuBLAS epilogue** on 192×192 FFN tiles: MFSDP's flat buffer produces output
     tensor sub-views with non-standard base pointers, preventing TMA epilogue (`tst`) → falls back
     to slower shared-memory epilogue (`tss`). ~175ms/step structural penalty.
   - **NCCL SM contention**: MFSDP's faster RING_LL (from NCCL flags) runs more aggressively,
     competing with cuBLAS for SMs → additional ~440ms/step GEMM slowdown vs no-flag run.

2. **AllGather −7.7%** (3,583ms → 3,309ms): Likely collective implementation or chunking differences.

3. **Wall time −3.3%** despite MFSDP's RS advantage: GEMM gap (−1,163ms/step) overwhelms the RS gain
   (+869ms/step), yielding net FSDP2 lead of 293ms/step.

### MFSDP Advantages

1. **ReduceScatter −25%** (3,482ms → 2,613ms): Driven by `NCCL_P2P_NET_CHUNKSIZE=2097152` and
   `NCCL_IB_SL=1`. RS avg duration 15,935µs vs FSDP2 21,494µs. **Note: this is an unfair advantage**
   — without these flags, MFSDP RS = FSDP2 RS exactly (see `analysis_report_no_flag.md`).

2. **Total comm −8.5%** (6,210ms vs 6,792ms): RS advantage outweighs the AG gap.

3. **Buffer copy overhead 96× less** (9ms vs 878ms/step): Flat buffer eliminates `chunk_cat`
   (476ms/step) and `split_with_sizes_copy_out` (402ms/step). Mostly overlapped but `chunk_cat`
   is serial before each RS, inflating FSDP2's measured RS latency.

4. **Lower memory footprint**: Peak allocated 46.2 vs 53.1 GB (−6.9 GB), reserved 61.59 vs 68.07 GB
   (−6.5 GB) per GPU. See §5.

---

## 3. Improvement Opportunities

### For Automodel FSDP2

1. **Tune NCCL settings** — `NCCL_P2P_NET_CHUNKSIZE=2097152` and `NCCL_IB_SL=1` gave MFSDP 33% RS
   speedup on this cluster. Testing for FSDP2 could reduce RS from 3,482ms → ~2,613ms/step, widening
   the FSDP2 wall-time lead from 3.3% to ~13%.

2. **Adopt contiguous flat parameter buffer** — Eliminates `chunk_cat` serial bottleneck (476ms/step
   before each RS) and `split_with_sizes_copy_out` (402ms/step post-AG). Even though mostly overlapped
   today, `chunk_cat` is a hard ceiling on RS latency improvement.

3. **Maintain AG scheduling quality** — FSDP2's AG advantage is a genuine strength. Ensure it is
   preserved in future refactors.

### For Megatron MFSDP

1. **Fix `tss` GEMM via output tensor alignment** — GEMM outputs into flat buffer offset views prevent
   TMA epilogue (`tst`). Fix: ensure output sub-views are 16-byte aligned with contiguous strides, or
   pre-allocate aligned staging buffers. Recovers ~175ms/step structural penalty (~2% wall time).

2. **Reduce NCCL SM contention** — The NCCL flags that give 33% RS speedup also cause ~440ms/step
   extra GEMM slowdown via SM competition. Investigate SM margin settings or scheduling to decouple
   the two effects.

3. **Add NVTX layer markers** — MFSDP profile has no module-level annotations, making per-layer
   scheduling analysis impossible.

---

## 4. Detailed Evidence

### 4.1 ReduceScatter Deep Dive

| Metric | MFSDP | FSDP2 |
|---|---|---|
| RS kernel | `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL` | Same |
| RS avg duration | **15,935 µs** | 21,494 µs |
| Min / p50 / Max | 12,455 / 15,895 / 19,646 µs | 17,205 / 21,665 / 25,962 µs |
| RS calls/step | 164 | 162 |
| RS total/step | **2,613ms** | 3,482ms |
| Winner | **MFSDP −25%** | |

The 33% per-call speedup is driven by `NCCL_P2P_NET_CHUNKSIZE=2097152` and `NCCL_IB_SL=1` — both
absent in FSDP2. Without these flags, MFSDP RS = FSDP2 RS exactly (21,491 vs 21,494µs, confirmed
in `analysis_report_no_flag.md`).

### 4.2 AllGather Detail

| Metric | MFSDP | FSDP2 |
|---|---|---|
| AG avg duration | 10,990 µs | **10,276 µs** |
| Min / Max | 5,721 / 14,709 µs | 167 / 23,906 µs |
| AG calls/step | 326 | 322 |
| AG total/step | 3,583ms | **3,309ms** |
| Winner | | **FSDP2 −7.7%** |

### 4.3 Full Kernel Breakdown per Step

| Category | MFSDP cnt/step | MFSDP ms/step | FSDP2 cnt/step | FSDP2 ms/step | Winner |
|---|---|---|---|---|---|
| AllGather | 326 | 3,583 | 322 | 3,309 | FSDP2 −7.7% |
| ReduceScatter | 164 | **2,613** | 162 | 3,482 | **MFSDP −25%** |
| GEMM total | 2,566 | 7,666 | 4,166 | 6,503 | FSDP2 −15.2% |
| Attention FWD (cuDNN) | 320 | 218 | 320 | 190 | FSDP2 −13% |
| Attention BWD (cuDNN) | 160 | 280 | 160 | 269 | Near-tie |
| RMSNorm FWD+BWD | 964 | ~72 | 964 | ~83 | Tie |
| Copy overhead (buffer) | 161 | **9** | ~962 | **878** | MFSDP 96× less |
| **Step wall time** | | **8,917** | | **8,624** | **FSDP2 −3.3%** |

Comm vs Compute:

| | MFSDP | FSDP2 |
|---|---|---|
| Comm (AG+RS) | **6,210ms** (41.6%) | 6,792ms (44.6%) |
| Compute (GEMM+attn) | 8,164ms (54.7%) | **6,962ms** (45.7%) |

### 4.4 GEMM Per-Shape Breakdown

> **Reading guide**: Compare **total ms/step**, not avg µs. Avg µs differs because the calls cover
> different problem sizes (see ³).

| Kernel shape | MFSDP ms/step | MFSDP avg µs | FSDP2 ms/step | FSDP2 avg µs | Notes |
|---|---|---|---|---|---|
| `tst_320x128_TNT` | 2,537 | 3,952 | 1,937 | 3,017 | Not apples-to-apples — see ³ |
| `tst_256x128_NNT` | 1,432 | 2,971 | 1,316 | 1,368 | FSDP2 2× calls (separate proj) |
| `tst_256x128_TNT` | 1,341 | 2,095 | 946 | 739 | FSDP2 2× calls (separate proj) |
| `tss_192x192_NTN` | 1,177 | **3,655** | 982 (`tst`) | **3,048** | MFSDP +20% — `tss` vs `tst` ⁴ |
| `tss_192x192_NTN` v2 | 540 | **3,378** | 507 (`tst`) | **3,167** | MFSDP +6.7% — `tss` vs `tst` |
| `tst_320x128_NNT` | 495 | 3,093 | 503 | 3,141 | Tie |
| **Total GEMM** | **7,666** | | **6,503** | | **FSDP2 −15.2%** |

³ **`tst_320x128_TNT`: different FLOPs per call — not a fair per-call comparison.**
For LLaMA 70B (GQA: 64 Q heads, 8 KV heads, head_dim=128):
- MFSDP: fused QKV weight `[8192, 10240]` → one GEMM, N=10240 FLOPs per call
- FSDP2: Q projection `[8192, 8192]` → `tst_320x128_TNT`, N=8192; K+V `[8192, 1024]` each → different
  smaller-tile rows

Both have ~642 calls in this row but cover different amounts of work. MFSDP's +31% per-call
(3,952 vs 3,017µs) is close to the expected N ratio (10240/8192 = 1.25×). Fused projections are
not causing the GEMM gap.

⁴ **`tss` vs `tst` — a major GEMM gap driver.** For 192×192 FFN tiles both sides run the same problem
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

This is a **direct structural consequence of MFSDP's flat buffer**. Contributes ~175ms/step to the
GEMM gap. An additional ~440ms/step comes from NCCL SM contention (see §4.1).

### 4.5 Attention Kernels

Both use identical cuDNN WGMMA kernels — fully fair comparison.

| | MFSDP | FSDP2 |
|---|---|---|
| FWD kernel | `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_4x` | Same |
| BWD kernel | `cudnn_generated_fort_native_sdpa_sm90_flash_bprop_wgmma_f16_knob_26_64x64x128_1x` | Same |
| FWD total/step | 218ms | **190ms** |
| BWD total/step | 280ms | **269ms** |

FSDP2 is slightly faster per call — likely HBM contention from concurrent AllGather streams in MFSDP
(more aggressive with NCCL flags active).

### 4.6 Buffer Copy Overhead

| Kernel | Role | FSDP2 ms/step | MFSDP ms/step |
|---|---|---|---|
| `chunk_cat_cuda_kernel` | Pre-RS: pack per-param grads into flat buffer | **476** | 0 |
| `split_with_sizes_copy_out` | Post-AG: unpack flat buffer back to param tensors | **402** | 0 |
| `CatArrayBatchedCopy` | MFSDP pre-AG shard copy from flat buffer | 0 | **9** |
| **Total** | | **878** | **9** |

FSDP2's `chunk_cat` (avg 2,937µs/call) is serial immediately before each RS — blocks RS launch.
The 878ms/step copy overhead is ~98% overlapped in wall time, but `chunk_cat` is a ceiling on RS
latency improvement.

### 4.7 On `NVTE_FWD/BWD_LAYERNORM_SM_MARGIN=16`

Set in MFSDP, not in FSDP2. Leaves 16 SMs idle during RMSNorm kernel launches, reserving them for
concurrent NCCL streams.

- **Effect on GEMM**: None. cuBLAS kernel selection is independent of this setting.
- **Effect on `tss` vs `tst`**: None. That is driven by output tensor memory layout (flat buffer
  base pointer alignment), not SM availability.
- **Wall-time impact**: ~5% slower RMSNorm per call → ~4ms/step; negligible in the overall comparison.

---

## 5. Memory Analysis

Source: Megatron `--log-memory-to-tensorboard` (MFSDP, rank 0) and Automodel per-iteration log
(FSDP2, rank 0). Steady-state values after warmup iteration.

### 5.1 Summary

| Metric | MFSDP | FSDP2 | Difference |
|---|---|---|---|
| **Peak allocated** | **46.2 GB** | **53.1 GB** | FSDP2 +6.9 GB (+15%) |
| **Reserved** | **61.59 GB** | **68.07 GB** | FSDP2 +6.5 GB (+10%) |
| Headroom (reserved − peak alloc) | 15.4 GB | 15.0 GB | ~tie |
| H100 capacity (80 GB) utilization | 77% | 85% | FSDP2 closer to limit |

**MFSDP uses ~6.5–7 GB less per GPU** across both metrics. Memory footprint is structural and
independent of NCCL flags.

### 5.2 Root Cause

**Peak allocated gap (6.9 GB)**: During AllGather, FSDP2 runs `chunk_cat` to pack individual parameter
tensors into a flat send buffer. At the AG peak, both the original per-parameter tensors and the packed
flat buffer are live simultaneously — two copies of the weight shards coexist briefly. MFSDP's
parameters already live in a flat contiguous buffer, so AllGather requires no temporary copy.

**Reserved gap (6.5 GB)**: PyTorch's caching allocator sizes its reserved blocks to the peak allocated
watermark. A lower peak allocated → smaller blocks acquired from CUDA → less reserved memory held.
The reserved gap tracks the allocated gap closely, confirming the cause is structural rather than
allocator fragmentation.

### 5.3 Implication

At 80 GB H100 SXM capacity, MFSDP's lower footprint (77% vs 85% utilization) leaves ~6.5 GB more
headroom per GPU. This margin could be used to:
- Increase MBS (micro-batch size) for higher GPU utilization
- Extend sequence length beyond 4096
- Enable longer activation checkpointing intervals (fewer recompute layers)
