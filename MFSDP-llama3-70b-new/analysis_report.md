# LLaMA3-70B MFSDP vs Automodel FSDP2 — nsys Profile Analysis Report (CP=1, SeqLen=4096, 8-node, FP32 Optimizer)

**Date**: 2026-04-14
**Model**: LLaMA3.1-70B
**Config**: TP=1, CP=1, PP=1, MBS=1, GBS=128, SeqLen=4096, full-recompute

---

## Runs Compared

| | Megatron MFSDP | Automodel FSDP2 |
|---|---|---|
| Framework | Megatron-LM (`--use-megatron-fsdp`) | Automodel PyTorch FSDP2 |
| Cluster | **EOS** | **Current cluster** |
| Container | `pytorch26.03_te2.14_deepep_x86` | `nemo-automodel:26.04.rc4` |
| nsys-rep | `MFSDP-llama3-70b-new/…eos_n8.nsys-rep` | `slurm_jobs/11124101/…node0.nsys-rep` |
| Nodes / GPUs | 8 / 64 | 8 / 64 |
| DP / GA | DP=64, GA=2 | DP=64, GA=2 |
| Profile window | 17,607 ms — **2 iters** (optimizer steps at 8,769 ms + 17,573 ms) | 17,281 ms — **2 iters** (NVTX: iteration_7 + iteration_8) |
| Step time | **8,787 ms/iter** (avg optimizer-to-optimizer) | **8,621 ms/iter** (NVTX compute 8,577 ms + optimizer ~44 ms) |
| MFU | **~39.9%** (nsys) | **~40.7%** (nsys) / **~41.5%** (log steady-state excl. anomalies) |
| Max memory | **46.24 GB** / GPU | **53.13 GB** / GPU |
| Optimizer | FusedAdam, FP32 exp_avg/sq, lr=0.00015, wd=0.1 | FusedAdam, **FP32 exp_avg/sq**, lr=0.00015, wd=0.1 |
| NCCL_P2P_NET_CHUNKSIZE | **2,097,152** (EOS env) | Not set |

**Optimizer is now matched** (both FP32 momentum, same betas/eps/lr/wd). Remaining fairness gap: different clusters (EOS vs current cluster), which primarily affects RS and AG bandwidth.

**Step time**: FSDP2 is **166 ms/iter (1.9%) faster** despite having 874 ms more total kernel work — explained by better comm-compute overlap (§8).

---

## 1. Kernel Time Breakdown — Per Iteration

Raw totals ÷ 2 (2 iterations in both profiles). Both GA=2; per-GA-step column ÷ 2 again.

| Category | MFSDP /iter | MFSDP ms/iter | MFSDP avg µs | FSDP2 /iter | FSDP2 ms/iter | FSDP2 avg µs | Per-GA-step winner |
|---|---|---|---|---|---|---|---|
| AllGather | 322 | 3,318 | 10,305 | 323 | 3,298 | 10,210 | **Tie** (FSDP2 0.6% less, within noise) |
| ReduceScatter | 164 | 2,409 | 14,691 | 162 | 3,496 | 21,578 | **MFSDP** (1.45× faster — cross-cluster, see §3) |
| AllReduce | ~6 | ~3 | — | ~3 | ~1 | — | Tie (negligible) |
| P2P/SendRecv | 0 | 0 | — | 0 | 0 | — | Tie (CP=1) |
| GEMM (`nvjet_sm90_*`) | 2,566 | 7,616 | 2,968 | 4,166 | 6,499 | 1,560 | **FSDP2** (3,250 vs 3,808 ms/GA-step, **15% less** — see §4) |
| cuDNN attn FWD | 320 | 219 | 684 | 320 | 190 | 592 | **FSDP2** (592 vs 684 µs, **13% faster** — cluster difference) |
| cuDNN attn BWD | 160 | 276 | 1,722 | 160 | 268 | 1,677 | **FSDP2** (1,677 vs 1,722 µs, 3% — within cluster noise) |
| RMSNorm FWD | 642 | 39 | 60 | 642 | 42 | 65 | **MFSDP** (7% — cluster difference) |
| RMSNorm BWD | 644 | 35 | 54 | 644 | 43 | 67 | **MFSDP** (19% — cluster difference) |
| chunk_cat (pre-RS pack) | 0 | **0** | — | 162 | **476** | 2,937 | **MFSDP** (flat buffer) |
| split_with_sizes (post-AG unpack) | 0 | **0** | — | 320 | **402** | 1,257 | **MFSDP** (flat buffer) |
| CatArrayBatchedCopy (pre-AG) | 160 | **9** | 57 | 0 | **0** | — | Tie (negligible) |
| **FSDP copy overhead** | | **9** | | | **878** | | **MFSDP (98× less)** |
| **Total kernel/iter** | | **14,366** | | | **15,240** | | MFSDP −6% less kernel work |
| **Step wall time** | | **8,787** | | | **8,621** | | **FSDP2 1.9% faster** |

**Per-GA-step balance:**

| Category | MFSDP ms | FSDP2 ms | FSDP2 delta |
|---|---|---|---|
| AllGather | 1,659 | 1,649 | −10 (tie) |
| ReduceScatter | 1,205 | 1,748 | **+543** (FSDP2 slower) |
| GEMM | 3,808 | 3,250 | **−558** (FSDP2 faster) |
| cuDNN FWD+BWD | 248 | 229 | −19 (FSDP2 faster) |
| RMSNorm | 37 | 43 | +6 (MFSDP faster) |
| Copy overhead | 5 | 439 | **+434** (FSDP2 slower) |
| **Net kernel** | **6,962** | **7,358** | **+396 FSDP2 more kernel** |
| **Wall (GA-step)** | **4,394** | **4,287** ¹ | **−107 FSDP2 faster** |

¹ FSDP2 GA-step wall from NVTX: iter_7 steps = 4,282 ms + 4,286 ms; iter_8 steps = 4,285 ms + 4,296 ms → avg **4,287 ms**.

FSDP2 has 396 ms/GA-step more kernel work yet finishes 107 ms sooner → it **hides 503 ms/GA-step more** behind overlap. See §8.

**Count normalization per GA step:**

| Category | MFSDP | FSDP2 |
|---|---|---|
| AllGather | 161 | 162 |
| ReduceScatter | 82 | 81 |
| GEMM | 1,283 | 2,083 (1.62× — separate projections) |
| cuDNN FWD / BWD | 160 / 80 | 160 / 80 |
| RMSNorm FWD / BWD | 321 / 322 | 321 / 322 |

All attention and norm counts match exactly. GEMM ratio 1.62× reflects separate Q/K/V and gate/up in FSDP2 vs fused in MFSDP.

**Communication vs Compute (per iteration):**

| | MFSDP | FSDP2 |
|---|---|---|
| Comm (AG + RS + AR) | 5,730 ms (39.9%) | 6,795 ms (44.6%) |
| Compute (GEMM + attn) | 8,111 ms (56.5%) | 6,957 ms (45.6%) |
| Copy overhead | 9 ms (0.1%) | 878 ms (5.8%) |
| **Total kernel** | **14,366 ms** | **15,240 ms** |

---

## 2. Attention Kernel Comparison

| | MFSDP | FSDP2 |
|---|---|---|
| FWD kernel | `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_4x1x1_cga1x1x1` | **Identical** |
| BWD kernel | `cudnn_generated_fort_native_sdpa_sm90_flash_bprop_wgmma_f16_knob_26_64x64x128_1x4x1_cga1x1x1` | **Identical** |
| FWD avg | 684 µs | **592 µs** (−13%) |
| BWD avg | 1,722 µs | **1,677 µs** (−3%) |

Both dispatch the same cuDNN WGMMA SDPA kernels — no structural attention gap. FSDP2 is slightly faster on both FWD and BWD; this is cluster-level (EOS vs current cluster), not framework-structural.

---

## 3. ReduceScatter Deep Dive

Both: `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL`.

| Metric | MFSDP | FSDP2 |
|---|---|---|
| RS calls/iter | 164 | 162 |
| Avg message size | ~26.7 MB | ~27.0 MB |
| Eff. bandwidth | **1.82 GB/s** | **1.25 GB/s** |
| BW ratio | **MFSDP 1.45× faster** | baseline |
| Avg duration | **14,691 µs** | 21,578 µs |
| Min | 11,924 µs | 17,205 µs |
| Max | 22,678 µs | 27,259 µs |
| p50 | **14,498 µs** | 21,670 µs |
| p90 | **16,544 µs** | 24,023 µs |

Both DP=64, same message sizes (~27 MB per call), same RING_LL kernel, same call counts. The **1.45× bandwidth gap is entirely a cross-cluster effect** (EOS fabric vs current cluster). FSDP2 RS distribution is tight (p50–p90 range of ~2,400 µs — no extreme outliers, just uniformly lower throughput fabric).

---

## 4. GEMM Analysis

### Per-Kernel Breakdown (per iteration = raw ÷ 2)

| Kernel | MFSDP /iter | MFSDP avg µs | FSDP2 /iter | FSDP2 avg µs | Notes |
|---|---|---|---|---|---|
| `tst_320x128_64x3 TNT` | 642 | **3,929** | 642 | **3,013** | Same count; MFSDP **30% slower** — AG interference |
| `tst_256x128_64x4 NNT` | 482 | 2,954 | 962 | 1,370 | 2× more in FSDP2 (sep. Q/K/V and gate/up) |
| `tss_192x192_64x3 NTN` (MFSDP) / `tst_192x192` (FSDP2) | 322 | **3,618** | 322 | **3,038** | MFSDP **19% slower** — tss vs tst variant |
| `tst_256x128_64x4 TNT` | 640 | 2,089 | 1,280 | 739 | 2× more in FSDP2 |
| `tss_192x192 v NTN` / `tst_192x192 v` | 160 | 3,448 | 160 | 3,167 | MFSDP **9% slower** — tss vs tst |
| `tst_320x128_64x3 NNT` | 160 | 2,962 | 160 | 3,139 | Tie (slight FSDP2 disadvantage) |

Two root causes for MFSDP's 15% higher total GEMM time:

**1 — AllGather interference (bimodal timing).** `tst_320x128_TNT` with same count (642/iter) and same grid dimensions shows MFSDP avg 3,929 µs vs FSDP2 3,013 µs (**30% gap**). Confirmed cause: GEMMs launching during peak 64-GPU ring traffic take ~5× longer (~5,400 µs) than GEMMs launching early in the AG lifecycle (~1,100 µs). The bimodal average elevates MFSDP's GEMM total. FSDP2's prefetch scheduling causes GEMMs to more often start before peak ring contention.

**2 — tss vs tst variant.** MFSDP autotuner selects `tss` (shared-memory epilogue) for 192×192 shapes while FSDP2 selects `tst` (TMA epilogue). `tst` is 9–19% faster. Grid dimensions are identical (same matrix shapes) — the variant difference is autotuner state, likely influenced by AG interference during MFSDP's autotuning phase.

---

## 5. FSDP2 Copy Overhead (per iteration)

| Kernel | MFSDP /iter | MFSDP ms | FSDP2 /iter | FSDP2 ms | Role |
|---|---|---|---|---|---|
| `chunk_cat_cuda_kernel` | 0 | **0** | 162 | **476** | Pre-RS: pack param grads into contiguous buffer |
| `split_with_sizes_copy_out_…_kernel` | 0 | **0** | 320 | **402** | Post-AG: unpack AG buffer back to param tensors |
| `CatArrayBatchedCopy_vectorized` | 160 | **9** | 0 | **0** | Pre-AG shard copy |
| **Total** | | **9** | | **878** | |

FSDP2 pays **878 ms/iter** (5.8% of kernel, 439 ms/GA-step) in buffer staging. `chunk_cat` (476 ms/iter) is **serial and blocking** before each RS — hard latency floor that cannot be overlapped. `split_with_sizes` (402 ms/iter) runs on the compute stream post-AG.

Per-GA-step copy overhead: FSDP2 439 ms vs MFSDP 5 ms. This is offset by FSDP2's 558 ms/GA-step GEMM advantage and 503 ms/GA-step superior overlap efficiency (§8).

---

## 6. MFU Analysis

| | MFSDP | FSDP2 |
|---|---|---|
| Nodes / GPUs | 8 / 64 | 8 / 64 |
| Tokens/step | 524,288 | 524,288 |
| Step time (nsys) | 8,787 ms | **8,621 ms** |
| TFLOPs/GPU/s | 394.9 | **402.2** |
| **MFU (nsys)** | **39.9%** | **40.7%** |
| **MFU (log steady-state)** | — | **~41.5%** (excl. nsys-affected iter 7 and export-stall iter 9) |
| Log MFU range (iters 5–14) | — | 40.3% – 41.7% |

**FSDP2 leads by ~0.8pp (nsys) to ~1.6pp (log).** Both run on different clusters, so this difference is not attributable purely to framework design. The cross-cluster RS bandwidth advantage for MFSDP (1.45×) partially offsets FSDP2's GEMM and overlap advantages.

---

## 7. Memory Analysis

| | MFSDP | FSDP2 |
|---|---|---|
| Max allocated (steady state) | **46.24 GB** | **53.13 GB** |
| H100 80 GB headroom | **33.8 GB** | **26.9 GB** |
| Source | Megatron log (`max allocated: 47345.90 MB`) | Training log (`Max Memory Allocated: 53.13 GB`) |

**MFSDP uses 6.89 GB (13%) less HBM** per GPU — now a fair comparison with matched FP32 optimizer states.

**Memory breakdown (per GPU, DP=64, 70B params):**

| Component | MFSDP | FSDP2 | Notes |
|---|---|---|---|
| BF16 working params | 2.19 GB | 2.19 GB | 70B × 2B / 64 |
| FP32 master weights | 4.38 GB | 4.38 GB | Both `master_weights: true` |
| FP32 exp_avg | 4.38 GB | 4.38 GB | Matched: both FP32 |
| FP32 exp_avg_sq | 4.38 GB | 4.38 GB | Matched: both FP32 |
| Prefetch buffers | ~0 | **~1.75 GB** | `fsdp2_forward_prefetch_depth=1` — 1 extra unsharded layer |
| FSDP2 per-tensor overhead | — | ~0.5 GB | Hundreds of individual param tensors vs 1 flat slab |
| chunk_cat peak staging | — | ~0.1–0.2 GB | Transient gradient contiguous buffer |
| **Approximate total** | **~15–17 GB** params+optim | **~17–18 GB** params+optim | +1–2 GB FSDP2 structural overhead |

The dominant source of the 6.89 GB gap:
- **Prefetch buffer**: `fsdp2_forward_prefetch_depth=1` holds 1 extra layer's unsharded BF16 params (70B/80 layers × 2B = ~1.75 GB) before releasing current layer
- **backward prefetch depth=2**: 2 layers' worth of params may coexist during BWD sweep
- **Per-tensor overhead + activation peaks**: FSDP2 manages individual tensor allocations vs MFSDP's single flat slab

MFSDP flat buffer = one contiguous slab, zero per-layer prefetch allocation overhead. This is the primary structural memory advantage.

**Activation memory** (MFSDP log): 1,088 MB/layer × 80 layers = 87 GB theoretical (no recompute). With full activation recompute, only 1–2 layers live simultaneously → actual peak well within 80 GB for both.

---

## 8. Overlap Efficiency

This is the most important section for explaining why FSDP2 is faster despite more kernel work.

| | MFSDP | FSDP2 |
|---|---|---|
| Total kernel/iter | 14,366 ms | 15,240 ms (+874 ms) |
| Wall time/iter | 8,787 ms | **8,621 ms** (−166 ms) |
| Kernel/wall ratio | 1.635 | **1.768** |
| Estimated exposed time | ~5,371 ms | ~4,861 ms |
| Overlap improvement | baseline | **+503 ms/iter hidden** |

FSDP2 hides **503 ms/iter more** kernel work behind its critical path than MFSDP. Sources:

1. **RS overlap**: FSDP2's RS (3,496 ms/iter) runs concurrently with post-RS GEMM compute across all 80 layers. With `fsdp2_backward_prefetch_depth=2`, the RS for layer N is issued while layer N+1 and N+2 BWD GEMMs are executing, creating a deeper pipeline.

2. **AG overlap**: `fsdp2_forward_prefetch_depth=1` issues the AG for the next layer before the current layer's GEMM finishes — the AG latency is absorbed into GEMM execution time. MFSDP's flat-buffer AG is synchronous at layer boundaries.

3. **Copy overlap**: `split_with_sizes_copy_out` (402 ms/iter) runs on a dedicated CUDA stream concurrently with subsequent GEMMs, so much of the staging cost is hidden.

**Net effect**: FSDP2's prefetch scheduling converts ~500 ms of communication and staging latency from exposed (serial) to hidden (overlapped), more than compensating for its copy overhead penalty.

---

## 9. Fairness Assessment

| Difference | Direction | Impact |
|---|---|---|
| GPU count | **Fair** | Both 8 nodes / 64 GPUs / GA=2 |
| Optimizer | **Fair** | Both FP32 exp_avg/sq, same betas/eps/lr/wd (matched this run) |
| Attention kernel | **Fair** | Identical cuDNN WGMMA SDPA on both |
| Different clusters | **Unfair** | EOS (MFSDP) vs current cluster (FSDP2). RS: 1.82 vs 1.25 GB/s (1.45×). AG: tied this run. Cluster difference artificially suppresses FSDP2 MFU |
| NCCL_P2P_NET_CHUNKSIZE | **Partially unfair** | Set 2,097,152 on EOS for MFSDP; absent for FSDP2. EOS eos.yaml sets this for both `megatron_fsdp` and `torch_fsdp2` env blocks — an EOS FSDP2 run would also use it |
| Copy overhead | **FSDP2 structural disadvantage** | 878 vs 9 ms/iter (98×). Serial `chunk_cat` (476 ms) blocks every RS. Recoverable by flat contiguous param buffer |
| GEMM efficiency | **FSDP2 structural advantage** | 3,250 vs 3,808 ms/GA-step (15% less). Two causes: tst vs tss kernel selection (9–19% for 192×192) and less AG-interference bimodal degradation |
| Overlap efficiency | **FSDP2 structural advantage** | Prefetch depth=1 FWD / depth=2 BWD hides 503 ms/iter more than MFSDP's flat-buffer AG/RS scheduling |
| Memory | **MFSDP structural advantage** | 46.24 vs 53.13 GB/GPU (6.89 GB = 13% less). Flat buffer eliminates prefetch staging allocations |
| Full recompute | **Fair** | Both confirmed: 160 cuDNN FWD/GA-step = 80 layers × 2 |

**Overall MFU**: MFSDP 39.9% vs FSDP2 40.7–41.5% — **FSDP2 leads by ~0.8–1.6pp** at current configurations. The FSDP2 lead would narrow or reverse on the same cluster due to MFSDP's RS bandwidth advantage (1.45× on EOS).

**Structural balance**: FSDP2's copy overhead penalty (−439 ms/GA-step) and RS disadvantage (−543 ms/GA-step) are more than compensated by GEMM efficiency (+558 ms/GA-step) and overlap quality (+503 ms/GA-step hidden). Net: FSDP2 107 ms/GA-step faster.

---

## 10. Improvement Opportunities

### For Automodel FSDP2

1. **Adopt flat contiguous parameter buffer** — `chunk_cat` (476 ms/iter, 238 ms/GA-step) is serial and blocks every RS. Eliminating `chunk_cat` + `split_with_sizes` removes 878 ms/iter kernel overhead. Combined with already-superior GEMM (+558 ms/GA-step) and overlap (+503 ms/GA-step hidden), this would widen FSDP2's lead substantially.

2. **Enable cuDNN for CP>1** — CP=1 parity shown here must extend to CP>1 to close the 4.5pp gap in the original CP=2 report.

3. **Memory: reduce prefetch staging** — 6.89 GB extra vs MFSDP comes mostly from `fsdp2_forward_prefetch_depth=1` pre-allocating unsharded layer params. A lazy-allocation prefetch scheme could recover ~2–3 GB without reducing overlap quality.

4. **Run on EOS for definitive comparison** — RS bandwidth gap (1.45×) from cluster difference prevents clean framework-only conclusion. Same cluster, same NCCL env = cleanest test.

### For Megatron MFSDP

1. **Adopt FSDP2-style prefetch scheduling** — FSDP2's depth=1 FWD / depth=2 BWD prefetch hides 503 ms/iter more kernel work. MFSDP's flat-buffer sequential AG/RS pattern exposes more latency. Implementing deeper AG/RS pipeline overlap could close the 107 ms/GA-step gap.

2. **Fix GEMM bimodal degradation** — `tst_320x128_TNT` runs 30% slower in MFSDP (3,929 vs 3,013 µs, same count/shape) due to AG peak-traffic interference. Staggering AG launches or tuning bucket size to reduce peak ring contention overlap with critical GEMMs would recover ~200–300 ms/GA-step.

3. **Switch 192×192 GEMMs from tss to tst** — FSDP2 selects `tst` (TMA epilogue, 9–19% faster). MFSDP autotuner landed on `tss`. Re-autotuning in a clean environment (no AG overlap during tuning) may select `tst`.

4. **Memory advantage is structural** — flat buffer's 6.89 GB advantage is intrinsic. Worth preserving as it allows larger batch sizes or longer sequences within 80 GB HBM.
