# Qwen3-MoE-30B MFSDP vs Automodel FSDP2 — nsys Profile Analysis (Job 11143651)

**Date**: 2026-04-15
**Model**: Qwen3-MoE-30B (A3B activated, 128 experts, top-8)
**Config**: TP=1, CP=1, PP=1, EP=8, DP=32, MBS=2, GBS=128, SeqLen=4096, 4 nodes (32×H100 SXM5), MoE-act recompute

**Note on prior analysis**: An earlier comparison using FSDP2 job 11139924 vs the same MFSDP profile
is documented in `analysis_report_automodel_vs_mfsdp_gbs128.md`. This report uses a newer FSDP2 run
(job 11143651, nightly container 2026-04).

---

## Runs Compared

| | Megatron MFSDP | Automodel FSDP2 |
|---|---|---|
| Framework | Megatron-LM (`--use-megatron-fsdp`) | Automodel PyTorch FSDP2 |
| Container | `pytorch26.03_te2.14_deepep_x86` | `nemo-automodel:nightly_202604` |
| Job | (local profile) | 11143651 |
| nsys-rep | `MFSDP-qwen3-moe/qwen3_30b_a3b_mfsdp_tp1_pp1_ep8_cp1_hsdp1_deepep_mbs2_gbs128_seqlen4096_cw_n4_force-balancing-recompute-moe-act.nsys-rep` | `slurm_jobs/11143651/nsys_qwen3_moe_30b_deepep_autonvtx_moe_act_recompute_repect_fp32_reducescatter.nsys-rep` |
| Profile window | 4,201ms (N_STEPS=2, no NVTX — inferred) | **Exact iter 8 NVTX window** (2,216.9ms, N_STEPS=1) |
| **Step time (log)** | **~2,100ms** (4,201ms ÷ 2) | **2,204ms** (log avg iters 10–14) |
| **MFU** | ~18.16% | ~17.3% |
| NCCL env flags | NCCL_IB_SL=1, NCCL_P2P_NET_CHUNKSIZE=2097152, NVTE_FWD/BWD_LAYERNORM_SM_MARGIN=16 | None |

---

## 1. Fairness & Shared Configuration

### What is the same (fair)

| Factor | MFSDP | FSDP2 |
|---|---|---|
| Hardware | 4-node H100 SXM5, same cluster | ← same |
| Model | Qwen3-MoE-30B, hidden=2048, 48 layers, 128 experts, top-8 | ← same (minor vocab diff: 152,064 vs 151,936) |
| Parallelism | TP=1, CP=1, PP=1, EP=8, total 32 GPUs | ← same |
| Batch config | GBS=128, MBS=2, SeqLen=4096, GAS=2 | ← same |
| Activation recompute | MoE activations only | ← same |
| Attention backend | TE FusedAttention (cuDNN WGMMA) | ← same kernel |
| Expert dispatch | DeepEP (P2P-based) | ← same |
| Gate | Force-balanced routing | ← same |
| RS dtype | FP32 (`ncclDevKernel_ReduceScatter_Sum_f32_RING_LL`) | ← same |

### What is different (potential unfairness)

| Factor | MFSDP | FSDP2 | Impact |
|---|---|---|---|
| **NCCL_IB_SL=1** | **Set** | Not set | Proven ~33% RS kernel speedup (LLaMA3-70B analysis) |
| **NCCL_P2P_NET_CHUNKSIZE=2097152** | **Set** | Not set | Contributes to RS speedup above |
| NVTE_FWD/BWD_LAYERNORM_SM_MARGIN=16 | Set | Not set | ~4ms/step RMSNorm SM reservation effect |
| DP sharding | HSDP: outer=4 no_shard, inner=8 sharded | Full 32-way ZeRO-3 | Structural — MFSDP AG/RS over inner-8 GPUs only, 4 concurrent groups |
| Parameter buffer | Flat contiguous buffer | Individual tensors per param | Drives `tss` vs `tst` GEMM epilogue; buffer copy overhead |
| `reshard_after_forward` (MoE) | Not set | **true** | Expert weights freed after forward; extra AG during backward expert recompute |
| Optimizer | DistributedOptimizer (overlapped param gather + grad reduce) | FusedAdam | MFSDP does extra param-recovery AG per step |
| Double buffer | `fsdp-double-buffer` enabled | Not set | MFSDP prefetches next GA step's params |
| NVTX annotations | None | AutoNVTX per module + `iteration_N_ga_step_M` | FSDP2 exact per-step analysis; MFSDP N_STEPS inferred |
| Container / NCCL | pytorch26.03, NCCL 2.29.7+cuda13.2 | nightly 2026-04, NCCL 2.29.2+cuda13.1 | Minor version differences |

### Window normalization

**MFSDP**: N_STEPS=2 assumed (no NVTX markers; 4,201ms ÷ ~2,100ms/step ≈ 2). Consistent with attention
count (192 FWD calls / 96 expected/step = 2 steps) and RMSNorm count (2,316 / 1,158/step = 2 steps).
All per-step values = raw ÷ 2.

**FSDP2**: Kernels are queried directly from the **exact iter 8 NVTX window** — no N_STEPS division
needed. NVTX timestamps:
```
iteration_8_ga_step_0: start = 2,323,154,119 ns  (min across 8 ranks)
iteration_8_ga_step_1: end   = 4,540,045,513 ns  (max across 8 ranks)
Window duration: 2,216.9 ms  = 1 complete optimizer step (2 GA steps)
```
All FSDP2 per-step values are raw kernel totals within this iter 8 window, device 0.

**Verification — RMSNorm total kernel count (iter 8 exact)**:

| | FSDP2 iter 8 (device 0) | MFSDP per-step (÷ 2) |
|---|---|---|
| Total RMSNorm (all types) | **1,158** | **1,158** |

Exact match → confirms iter 8 represents exactly 1 optimizer step, consistent with MFSDP N_STEPS=2.

---

## 2. High-Level Results & Adv / Disadv

### Headline

**MFSDP is ~4.7% faster** (~2,100ms vs 2,204ms/step), but holds active NCCL flags that give it a
known RS kernel advantage. The kernel-level story is nuanced: FSDP2 has dramatically faster comm
kernels (AG −24%, RS −57%) but wastes nearly all AG savings because **FSDP2 AG is 98.3% exposed**
(588ms/step lost vs MFSDP's 311ms). MFSDP also has lower total GEMM (646ms vs 745ms, despite `tss`
epilogue) — FSDP2's `reshard_after_forward: true` adds extra expert GEMMs during backward recompute.
The two effects together keep MFSDP's total exposed comm lower (624ms vs 731ms), preserving its
step-time lead.

| Metric | MFSDP (÷ n=2) | FSDP2 (iter 8 exact) | Winner |
|---|---|---|---|
| **Step wall time (log)** | **~2,100ms** | **2,204ms** | **MFSDP −4.7% (w/ NCCL flags)** |
| MFU | ~18.2% | ~17.3% | — |
| AG kernel time/step | 789.9ms | **599.0ms** | **FSDP2 −24.2%** |
| RS kernel time/step | 721.4ms | **310.6ms** | **FSDP2 −57.0%** |
| Total comm kernel/step | 1,521.8ms | **909.6ms** | **FSDP2 −40.2%** |
| **GEMM total/step** | **646.3ms** | 744.6ms | **MFSDP −13.2%** |
| Attention FWD+BWD/step | 174.8ms | **169.2ms** | Near-tie (FSDP2 −3.2%) |
| DeepEP dispatch+combine/step | 432.9ms | **213.9ms** | **FSDP2 −50.6%** |
| Buffer copy overhead/step | ~4ms | ~76ms | MFSDP 19× less |
| **AG truly exposed/step** | **311ms** | 588ms | **MFSDP −47.1%** |
| **RS truly exposed/step** | 313ms | **143ms** | FSDP2 −54.3% |
| **Total truly exposed comm/step** | **624ms** | 731ms | **MFSDP −14.6%** |

### FSDP2 Advantages

1. **AG kernel −24%** (599.0ms vs 789.9ms/step): FSDP2's single 32-way ZeRO-3 AG stream has less
   total kernel time per step. MFSDP has 2 AG streams (weight AG + param-recovery AG from
   DistributedOptimizer), inflating total AG kernel time.

2. **RS kernel −57%** (310.6ms vs 721.4ms/step): FSDP2 RS is substantially faster at kernel level.
   However, MFSDP has active `NCCL_IB_SL=1` + `NCCL_P2P_NET_CHUNKSIZE=2097152` — proven to give ~33%
   RS kernel speedup in LLaMA3-70B. Without those flags, MFSDP RS ≈ 1,075ms/step (gap widens to ~3.5×).
   MFSDP also performs 2× more RS calls (392 vs 198 per 2-step window) from HSDP's inner 8-way RS and
   the DistributedOptimizer's gradient reduce passes.

3. **DeepEP dispatch+combine −51%** (213.9ms vs 432.9ms/step): FSDP2 dispatch avg 292µs vs MFSDP
   929µs (3.2× faster per call); combine avg 282µs vs MFSDP 1,209µs (4.3× faster per call). FSDP2
   uses a newer DeepEP version with `notify_dispatch` (102µs) and `cached_notify_dispatch` (378µs)
   pipelining variants. MFSDP has older dispatch (929µs) and combine (1,209µs) kernels.

4. **All GEMM `tst` epilogue**: Zero `tss` kernels in FSDP2. Expert and dense GEMMs all select
   TMA-store epilogue; fixing MFSDP's `tss` issue would close most of the GEMM gap.

### MFSDP Advantages
1. **Lower total GEMM −13%** (646ms vs 745ms/step): Counterintuitively, despite 3,362 calls/step
   using the slow `tss` shared-memory epilogue (~223ms/step penalty), MFSDP total GEMM is still lower
   than FSDP2. 
   If MFSDP's `tss` is
   fixed (saving ~116ms/step, see §4.2), MFSDP GEMM would be ~530ms — 29% better than FSDP2 (745ms).

2. **Better AG comm/compute overlap — exposed AG 47% lower**: Despite 32% more AG kernel time, MFSDP
   hides 53% of AG (vs 1.7% for FSDP2), yielding 311ms vs 588ms exposed AG per step. The slow
   DeepEP dispatch/combine inadvertently provides compute cover for AG — the very bottleneck that
   hurts step time also serves as AG overlap.

3. **FSDP2 AG is 98.3% exposed**: Only 8/494 FSDP2 AG kernel invocations are even partially covered
   by any compute. Expert GEMMs run on dedicated CUDA streams (171–174) that the AG prefetch scheduler
   does not treat as valid compute overlap — AG fires during the main stream's idle gap between
   attention and expert dispatch.

4. **Lower total exposed comm −15%** (624ms vs 731ms/step): Better AG overlap more than compensates
   for FSDP2's better RS overlap.

5. **Buffer copy 19× less** (~4ms vs ~76ms/step): Flat buffer eliminates `chunk_cat` pre-RS packing
   and `split_with_sizes_copy_out` post-AG unpacking. Narrower than LLaMA3-70B (96× less) because
   MoE models have fewer dense sharded parameters.

6. **NCCL tuning upside**: Active flags give the current step-time edge.

---

## 3. Improvement Opportunities

### For Automodel FSDP2

1. **Fix AG prefetch overlap for MoE** (highest priority) — FSDP2 AG is 98.3% exposed (588ms/step
   wasted). If fixed to match MFSDP's 46.8% exposure, exposed AG drops from 588ms → ~281ms/step —
   saving ~307ms/step, bringing step time from 2,204ms toward ~1,897ms (MFU: ~17.3% → ~20.3%).
   Fix: register expert GEMM streams (streams 171–174) as valid overlap partners for AG scheduling.

2. **Evaluate `reshard_after_forward: true` cost** — Adds 36% more GEMM operations per step
   (13,446 vs 9,894 for MFSDP). These extra GEMMs cost ~99ms/step net overhead (745ms vs 646ms MFSDP
   GEMM despite MFSDP having tss penalty). Profile without this flag to quantify net cost vs memory
   benefit; the extra AGs triggered (during backward) are also 98% exposed.

3. **Contiguous flat parameter buffer** — Eliminate `chunk_cat` (~15ms/step) and
   `split_with_sizes_copy_out` (~28ms/step). Currently ~76ms/step total; serial `chunk_cat` before RS
   blocks RS launch even when mostly overlapped.

### For Megatron MFSDP

1. **Fix `tss` GEMM epilogue** (~116ms/step improvement) — Flat buffer output views prevent TMA
   epilogue on 3,362 calls/step. Affected: `tss_128x192_NTN` (80.8ms/step), `tss_96x128_NTN`
   (77.4ms/step), `tss_256x128_NTT` (36.6ms/step), `tss_128x160_NTT` (26.9ms/step). If fixed,
   MFSDP GEMM drops from 646ms to ~530ms/step — 29% better than FSDP2's 745ms. Same fix as
   identified in LLaMA3-70B analysis.

2. **Optimize DeepEP dispatch/combine** (~219ms/step gap) — MFSDP dispatch avg 929µs vs FSDP2 292µs;
   combine avg 1,209µs vs FSDP2 282µs. Adopt `cached_notify_dispatch` pipelining from FSDP2's DeepEP.
   Caution: fixing dispatch/combine also reduces AG hiding (slow dispatch currently covers 41% of AG);
   pair this fix with an AG overlap fix to avoid regression.

3. **Add NVTX layer markers** — Without NVTX, N_STEPS must be inferred; per-layer AG/RS scheduling
   is invisible. Critical for debugging why RS is 43% exposed despite double-buffer.

4. **Evaluate NCCL flag sensitivity** — `NCCL_IB_SL=1` + `NCCL_P2P_NET_CHUNKSIZE` deliver the
   step-time advantage but are cluster-specific and known to hurt on EOS. Report clean baselines.

---

## 4. Detailed Evidence

### 4.1 Full Kernel Breakdown per Step

MFSDP values = raw ÷ 2. FSDP2 values = exact iter 8 NVTX window.

| Category | MFSDP cnt/step | MFSDP ms/step | FSDP2 cnt/step | FSDP2 ms/step | Winner |
|---|---|---|---|---|---|
| AllGather | 392 | 789.9 | 346 | 599.0 | FSDP2 −24.2% |
| ReduceScatter | 196 | 721.4 | 99 | 310.6 | FSDP2 −57.0% |
| AllReduce | 16.5 | 10.5 | — | — | MFSDP only (minor) |
| GEMM (nvjet tst) | 6,532 | 423.1 | 13,446 | 744.6 | MFSDP (tst only) |
| GEMM (nvjet tss) | 3,362 | 223.2 | 0 | 0 | FSDP2 (no tss) |
| **GEMM total** | **9,894** | **646.3** | **13,446** | **744.6** | **MFSDP −13.2%** |
| Attention FWD (cuDNN) | 96 | 45.5 | 96 | 43.6 | Near-tie |
| Attention BWD (cuDNN) | 96 | 129.3 | 96 | 125.6 | Near-tie |
| DeepEP dispatch | 192 | 178.4 | 288 | 84.3 | FSDP2 −52.8% |
| DeepEP combine | 192 | 232.2 | 192 | 54.2 | FSDP2 −76.7% |
| DeepEP notify_dispatch | 96 | 1.7 | 192 | 19.6 | — |
| DeepEP cached_notify | 96 | 2.5 | 96 | 36.3 | MFSDP (much faster kernel) |
| DeepEP layout | 96 | 6.4 | 192 | 12.9 | Near-tie |
| DeepEP cached_notify_combine | 192 | 11.6 | 192 | 6.5 | — |
| **DeepEP total** | **864** | **432.9** | **1,152** | **213.9** | **FSDP2 −50.6%** |
| Permute + unpermute | 384 | 63.0 | 576 | 61.0 | Near-tie |
| Buffer copy (chunk_cat + split) | ~0 | ~4 | ~924 | ~76 | MFSDP 19× less |
| RMSNorm (all types) | 1,158 | 44.5 | 1,158 | 46.9 | Near-tie |
| Elementwise | ~6,024 | 82.9 | 4,463 | 82.9 | Near-tie |
| Triton | 384 | 31.0 | 384 | 49.5 | MFSDP |
| **Step wall time (log)** | | **~2,100** | | **2,204** | **MFSDP −4.7%** |

Comm vs Compute:

| | MFSDP | FSDP2 |
|---|---|---|
| Comm (AG+RS+AR) | 1,521.8ms | 909.6ms |
| Compute (GEMM+attn) | 821.1ms | 913.8ms |

### 4.2 GEMM Epilogue Analysis (tss vs tst)

MFSDP `tss` (shared-memory epilogue, slower) kernels per 2-step window (→ per step = ÷2):

| Kernel | Count/2-step | Total/2-step | Avg | FSDP2 analogue |
|---|---|---|---|---|
| `nvjet_sm90_tss_128x192_64x5_2x1_v_bz_coopB_NTN` | 3,072 | 161.5ms | 52.6µs | → `tst` |
| `nvjet_sm90_tss_96x128_64x6_2x1_v_bz_NTN` | 3,072 | 154.9ms | 50.4µs | → `tst` |
| `nvjet_sm90_tss_256x128_64x4_1x2_h_bz_coopA_NTT` | 196 | 73.3ms | 373.9µs | → `tst` |
| `nvjet_sm90_tss_128x160_64x5_2x1_v_bz_NTT` | 192 | 53.7ms | 279.8µs | → `tst` |
| **Total `tss`** | **6,532** | **443.4ms** | | **~223ms/step** |

FSDP2: **zero `tss` kernels**. Root cause identical to LLaMA3-70B: MFSDP GEMM outputs into offset views
of the flat parameter/gradient buffer — non-standard base pointer alignment prevents cuBLAS from
constructing a valid TMA tile descriptor → falls back to `tss`.

**tss speedup estimate**: tss kernels avg 52–54µs for small tiles. Equivalent tst tile would run at ~32µs
(based on similar FSDP2 tst kernels). If fixed: 3,362 calls × (52–32)µs ≈ **~116ms/step savings**.
After fix: MFSDP GEMM → 646 − 116 = **~530ms/step** vs FSDP2 745ms → MFSDP 29% better on GEMM.

**Why FSDP2 GEMM is higher (745ms) despite all-tst**: `reshard_after_forward: true` triggers extra
expert weight re-gathers during backward, causing FSDP2 to run 13,446 GEMM calls/step vs MFSDP's 9,894
(+36% more calls). Each FSDP2 GEMM call is also larger on average (55µs vs ~32µs MFSDP average)
consistent with more expert batching per gather.

### 4.3 AllGather Detail

| Metric | MFSDP | FSDP2 (iter 8) |
|---|---|---|
| AG streams | 2 (stream 35 weight AG; stream 31 param-recovery) | 1 (stream 28) |
| AG calls/step | 392 | 346 |
| AG total kernel time/step | 789.9ms | 599.0ms |
| AG avg duration per call | 2,014.9µs | 1,731µs |
| AG merged wall/step | 665.2ms | 599.0ms |
| AG exposed — 3-bucket C/step | **311ms (46.8%)** | 588ms (98.3%) |
| AG exposed — sweep-line/step | 526ms (66.6%) | ~598ms (99.8%) |

MFSDP's 2nd AG stream (stream 31, 196 calls/step, ~179ms/step) handles param-recovery AG from
DistributedOptimizer `--overlap-param-gather` — no equivalent in FSDP2.

FSDP2 AG overlap breakdown (per-step, from 3-bucket analysis):
- A. Hidden by GEMM+attn: 0.0ms (0.0%)
- B. Hidden by light compute: 10.2ms (1.7%)
- C. Truly exposed: **588ms (98.3%)**

Root cause: expert GEMMs run on dedicated streams 171–174 which the AG prefetch scheduler does not
treat as compute overlap. AG fires during the main stream's dead period between attention and expert
dispatch.

### 4.4 ReduceScatter Detail

| Metric | MFSDP | FSDP2 (iter 8) |
|---|---|---|
| RS calls/step | 196 | 99 |
| RS total kernel time/step | 721.4ms | 310.6ms |
| RS avg duration per call | 3,680.5µs | 3,138µs |
| RS RING_LL algorithm | `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL` | Same |
| RS truly exposed/step | 312.8ms (43.4%) | ~143ms (46.0%) |

**NCCL flag impact**: `NCCL_IB_SL=1` + `NCCL_P2P_NET_CHUNKSIZE=2097152` active. From LLaMA3-70B
controlled experiment: 33% RS kernel speedup. Without flags, MFSDP RS avg ≈ 5,483µs/call →
total ≈ 1,075ms/step (gap vs FSDP2 widens from 2.3× to ~3.5×). MFSDP has 2× more RS calls (392 vs
198/2-step) from HSDP topology: inner 8-way RS per group + DistributedOptimizer grad-reduce passes.

### 4.5 DeepEP Dispatch and Combine

| Kernel | MFSDP count/step | MFSDP total/step | MFSDP avg | FSDP2 count | FSDP2 total | FSDP2 avg | Speedup |
|---|---|---|---|---|---|---|---|
| `dispatch` | 192 | 178.4ms | 929µs | 288 | 84.3ms | 292µs | FSDP2 3.2× |
| `combine` | 192 | 232.2ms | 1,209µs | 192 | 54.2ms | 282µs | FSDP2 4.3× |
| `notify_dispatch` | 96 | 1.7ms | 18µs | 192 | 19.6ms | 102µs | MFSDP (cheap notify) |
| `cached_notify_dispatch` | 96 | 2.5ms | 26µs | 96 | 36.3ms | 378µs | MFSDP (cheap) |
| `layout` | 96 | 6.4ms | 67µs | 192 | 12.9ms | 67µs | Similar per-call |
| `cached_notify_combine` | 192 | 11.6ms | 60µs | 192 | 6.5ms | 34µs | FSDP2 |
| **Total** | **864** | **432.9ms** | | **1,152** | **213.9ms** | | **FSDP2 2.0×** |

The 219ms/step dispatch+combine gap is one of the dominant MFSDP bottlenecks (alongside `tss` GEMM at
223ms/step). FSDP2's main kernel speedup is in `dispatch` (3.2×) and `combine` (4.3×), suggesting a
significantly different implementation of the token-routing CUDA kernel. MFSDP's notify/cached_notify
kernels are actually faster per-call (18–26µs vs 102–378µs in FSDP2), but the dominant
dispatch/combine kernels are 3–4× slower.

Note: FSDP2 has more dispatch calls per step (288 vs 192) because `reshard_after_forward: true`
triggers additional dispatch operations during the backward recompute.

### 4.6 Buffer Copy Overhead

| Kernel | Role | FSDP2 ms/step | MFSDP ms/step |
|---|---|---|---|
| `chunk_cat_cuda_kernel` | Pre-RS: pack per-param grads | ~14.8 | 0 |
| `split_with_sizes_copy_out_...` | Post-AG: unpack flat buffer → param tensors | ~27.6 | 0 |
| `CatArrayBatchedCopy` | Misc buffer ops | ~33.6 | ~4 |
| **Total** | | **~76ms** | **~4ms** |

Narrower than LLaMA3-70B (~843ms/step) because Qwen3-MoE has far fewer dense sharded parameters
(expert weights handled by EP independently of ZeRO).

### 4.7 Attention Kernels

| | MFSDP | FSDP2 (iter 8) |
|---|---|---|
| FWD kernel | `cudnn_...sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_...` | Same |
| BWD kernel | `cudnn_...sdpa_sm90_flash_bprop_wgmma_f16_knob_26_64x64x128_...` | Same |
| FWD calls/step | 96 | 96 |
| BWD calls/step | 96 | 96 |
| FWD total/step | 45.5ms | 43.6ms |
| BWD total/step | 129.3ms | 125.6ms |

Both have 96 attention calls/step = 48 layers × 2 GA-steps. Counts match as expected for MoE-act
recompute (attention FWD is re-run once during backward recompute per GA step). This symmetry
confirms the N_STEPS normalization is correct for both profiles.

### 4.8 Comm–Compute Overlap Analysis

Source: `analyze_comm_compute_overlap.py`, 3-bucket decomposition.
MFSDP values ÷ N_STEPS=2. FSDP2 percentages from full-window analysis (structural pattern holds for
iter 8); absolute values computed as percentage × iter 8 kernel time.

| Bucket | MFSDP AG | FSDP2 AG | MFSDP RS | FSDP2 RS |
|---|---|---|---|---|
| Comm merged wall/step | 665.2ms | 599.0ms | 721.4ms | 310.6ms |
| **A. Hidden by GEMM+attn** | 12.1% | **0.0%** | 31.3% | 37.7% |
| **B. Hidden by light compute** | 41.1% | 1.7% | 25.4% | 16.3% |
| **C. Truly exposed** | **46.8% (311ms)** | **98.3% (588ms)** | **43.4% (313ms)** | **46.0% (143ms)** |
| Total hidden (A+B) | 53.2% | 1.7% | 56.6% | 54.0% |

**Total truly exposed comm per step:**

| | MFSDP | FSDP2 |
|---|---|---|
| AG exposed | **311ms** | 588ms |
| RS exposed | 313ms | **143ms** |
| **Total exposed** | **624ms** | 731ms |
| **Winner** | **MFSDP −14.6%** | |

**Key observations:**

1. **FSDP2 AG overlap is not enabled for MoE (98.3% exposed)**: Categorically different from LLaMA3-70B
   (93.6% hidden there). Expert GEMMs on streams 171–174 are not recognized as compute overlap by the
   AG prefetch scheduler. AG fires during the main stream's dead period between attention and expert
   dispatch. Fix: register expert GEMM streams as valid overlap targets.

2. **MFSDP's slow dispatch+combine inadvertently hides AG**: Bucket B = 41.1% (273ms/step) from
   Triton ops, elementwise, and "other light" compute dominated by dispatch-adjacent kernels. The very
   bottleneck that hurts step time simultaneously serves as AG cover — fixing dispatch without also
   fixing AG overlap would likely worsen MFSDP's total exposed comm.

3. **RS overlap similar**: MFSDP 56.6% hidden vs FSDP2 54.0% hidden — both overlap RS effectively.
   MFSDP's larger absolute RS exposure (313ms vs 143ms) is proportional to RS kernel time — not a
   scheduling gap.

4. **MFSDP total exposed comm wins by 107ms** vs FSDP2: Despite having 40% more total comm kernel
   time, MFSDP's superior AG hiding yields less total exposed comm (624ms vs 731ms). This is a key
   reason MFSDP is faster overall despite worse comm kernels.

5. **Step-time math**: MFSDP total exposed (624ms) vs FSDP2 total exposed (731ms) = −107ms
   advantage. FSDP2 faster on: DeepEP (−219ms), buffer copy (−72ms), RS kernel difference (−411ms
   of which ~267ms is NCCL flags). MFSDP faster on: GEMM (−99ms), AG hiding (−277ms). Net math is
   complex but direction aligns with observed −104ms MFSDP lead.

---

## 5. Memory

| Metric | MFSDP | FSDP2 |
|---|---|---|
| **Peak allocated** | **~62.5 GB** (prior analysis estimate) | **60.84 GB** |
| H100 capacity | 80 GB | 80 GB |
| Utilization | ~78% | ~76% |
| **Difference** | | **FSDP2 −2.7%** |

FSDP2 uses slightly less memory.

---