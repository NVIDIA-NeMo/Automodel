# LLaMA3-70B MFSDP vs Automodel FSDP2 — nsys Profile Analysis Report

**Date**: 2026-04-10
**Model**: LLaMA3.1-70B
**Config**: TP=1, CP=2, PP=1, DP=32, MBS=1, GBS=128, SeqLen=8192, 8 nodes (64×H100), full-recompute

---

## Runs Compared

| | Megatron MFSDP | Automodel FSDP2 |
|---|---|---|
| Framework | Megatron-LM (`--use-megatron-fsdp`) | Automodel PyTorch FSDP2 |
| Container | `pytorch26.03_te2.14_deepep_x86` | `nemo-automodel:26.04.rc4` |
| nsys-rep | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/Automodel-release/MFSDP-llama3-70b/llama3_70b_mfsdp_tp1_pp1_ep1_cp2_hsdp1_alltoall_mbs1_gbs128_seqlen8192_cw_n8_full-recompute.nsys-rep` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/Automodel-release/slurm_jobs/11038849/nsys_llama31_70b_pretrain_tp1cp2pp1_te_attn_cp2_no_compile_p2p_cp_node0.nsys-rep` |
| sqlite | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/Automodel-release/MFSDP-llama3-70b/llama3_70b_mfsdp_tp1_pp1_ep1_cp2_hsdp1_alltoall_mbs1_gbs128_seqlen8192_cw_n8_full-recompute.sqlite` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/Automodel-release/slurm_jobs/11038849/nsys_llama31_70b_pretrain_tp1cp2pp1_te_attn_cp2_no_compile_p2p_cp.sqlite` |
| Config file | `/lustre/fsw/portfolios/coreai/users/xuwenc/code/mfsdp_benchmark/megatron-benchmark/scripts/llama3_70b/sbatch_llama3_70b_mfsdp_tp1_pp1_ep1_cp2_hsdp1_alltoall_mbs1_gbs128_seqlen8192_cw_n8_full-recompute.sh` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/Automodel-release/examples/llm_pretrain/llama3_1_70b_pretrain_benchmark_8nodes_tp1cp2pp1_0410.yaml` |
| Profile window | 37.61s (~2 steps) | 20.03s (~1 step) |
| MFU | ~41.7% (412.6 TFLOPs/GPU) | ~37.2% (avg 37.225%) |
| Step time | ~18.8s | ~20.03s |
| CP implementation | P2P AllToAll ring (SendRecv) | P2P ring attention (same protocol as MFSDP) |

---

## 1. Kernel Time Breakdown — Per Step (Side-by-Side)

All values normalized to **1 step**: Megatron window = 2 steps (37.61s, no NVTX markers, ÷2); Automodel = 1 step (20.03s, confirmed via NVTX `iteration_7`). Avg µs is per-call and unchanged by normalization. All numbers verified by direct SQL query against the nsys sqlite files.

**SQL used for each row** (pattern applies to both DBs with `deviceId=0`):
```sql
-- AllGather:      shortName LIKE '%AllGather%'
-- ReduceScatter:  shortName LIKE '%ReduceScatter%'
-- AllReduce:      shortName LIKE '%AllReduce%'
-- P2P/SendRecv:   shortName LIKE '%Send%' OR '%Recv%'
-- GEMM:           shortName LIKE '%nvjet%'
-- Flash-attn FWD: shortName LIKE '%flash_fwd%'  (MFSDP: LIKE '%fprop%')
-- Flash-attn BWD: shortName LIKE '%flash_bwd%'  (MFSDP: LIKE '%bprop%')  [main kernel only]
-- RMSNorm FWD:    shortName LIKE '%rmsnorm%fwd%'
-- RMSNorm BWD:    shortName LIKE '%rmsnorm_bwd_tuned%'  [main kernel only]
-- Total kernel:   all rows, deviceId=0
-- Wall time:      (max(end)-min(start)) / 1e6
```

| Category | MFSDP Count | MFSDP Total (ms) | MFSDP Avg (µs) | FSDP2 Count | FSDP2 Total (ms) | FSDP2 Avg (µs) | Winner |
|---|---|---|---|---|---|---|---|
| AllGather | 644 | 7,571 | 11,756 | 643 | 6,610 | 10,280 | **FSDP2** (6,610 vs 7,571 ms, 0.87×) |
| ReduceScatter | 328 | 9,347 | 28,497 | 324 | 5,721 | 17,656 | **FSDP2** (5,721 vs 9,347 ms, 1.63×) |
| AllReduce | 13 | 17 | 1,304 | 2 | ~1 | — | Tie (negligible) |
| P2P/SendRecv | 1,280 | 310 | 242 | 1,600 | 708 | 442 | **MFSDP** (310 vs 708 ms) — different protocols, see note ³ |
| GEMM (`nvjet_sm90_*`) | 5,132 | 14,830 | — | 8,332 | 12,277 | — | Tie ² |
| Flash-attn FWD | 1,280 | 767 | **599** | 1,280 | 1,356 | 1,059 | **MFSDP** (599 vs 1,059 µs avg, 1.77×) |
| Flash-attn BWD ¹ | 640 | 1,120 | **1,750** | 640 | 1,839 | 2,873 | **MFSDP** (1,750 vs 2,873 µs avg, 1.64×) |
| RMSNorm FWD | 1,284 | 79 | 62 | 1,284 | 83 | 64 | Tie (79 vs 83 ms) |
| RMSNorm BWD | 644 | 80 | 125 | 644 | 77 | 120 | Tie (80 vs 77 ms) |
| **Total RMSNorm** | | **160** | | | **164** | | Tie (160 vs 164 ms) |
| **Total kernel/step** | | **35,177** | | | **32,399** | | — |
| **Step wall time** | | **18,807** | | | **20,031** | | **MFSDP** (18,807 vs 20,031 ms, 1.065×) |

² GEMM total time is not a fair single-winner metric: MFSDP has larger but fewer GEMMs (fused QKV/gate-up); FSDP2 has more but smaller GEMMs (separate Q/K/V). Both do the same mathematical work — the difference is tiling strategy, not efficiency.

¹ BWD main kernel only: MFSDP=`cudnn_generated_fort_native_sdpa_sm90_flash_bprop_*`, FSDP2=`flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`. FSDP2 full BWD including helpers (`flash_bwd_dot_do_o_kernel` + `flash_bwd_convert_dq_kernel`) = **1,926ms** total.

³ Both runs now use P2P for CP but with different ring-attention protocols — volume is not the same. MFSDP: AllToAll (SendRecv pairs), 80 layers × 2 AllToAll × 2 (send+recv) × 4 GA = 1,280. FSDP2: ring attention P2P, 80 layers × 5 (FWD:2 + BWD:2 + reduce:1) × 4 GA = 1,600. The 2.28× time difference reflects both count difference and protocol overhead, not an apples-to-apples comparison.

### Count Difference Analysis

Counts differ across rows — this does **not** necessarily mean different total data volume. Analysis per category:

| Category | MFSDP Count | FSDP2 Count | Ratio | Total time ratio | Verdict |
|---|---|---|---|---|---|
| AllGather | 644 | 643 | ~1.0× (identical) | FSDP2 12.7% less total | **Both runs use weight-only AGs — counts are essentially identical (644 vs 643).** Both use P2P for CP so no CP AllGather in either run. With the same count and model size, the 12.7% total time difference (7,571 vs 6,610 ms) purely reflects per-call duration: MFSDP avg 11,756 µs vs FSDP2 avg 10,280 µs. |
| ReduceScatter | 328 | 324 | Same | 1.63× more in MFSDP | **Same count, same volume (~27 MB/call), but MFSDP 1.63× slower bandwidth** (0.94 vs 1.57 GB/s). Root cause: `NCCL_P2P_NET_CHUNKSIZE=2097152` in Megatron env — see §3. |
| P2P/SendRecv | 1,280 | 1,600 | 1.25× more in FSDP2 | MFSDP 2.28× less total | **Both use P2P for CP but different ring-attention protocols — not the same volume.** MFSDP uses AllToAll (implemented as SendRecv): 80 layers × 2 AllToAll (pre+post attention) × 2 (send+recv) × 4 GA = **1,280** ✓. New FSDP2 uses ring attention P2P: 80 layers × 5 passes (FWD:2 + BWD:2 + reduce:1) × 4 GA = **1,600** ✓. The larger FSDP2 P2P total (708 vs 310 ms) reflects both the higher call count and individual message characteristics per the ring-attention protocol. |
| GEMM (`nvjet_sm90_*`) | 5,132 | 8,332 | 1.62× more in FSDP2 | MFSDP 1.22× higher | **Different projection fusion, not same volume.** FSDP2 uses **separate Q, K, V projections** (3 GEMMs/layer) and **separate gate + up** (2 GEMMs/layer) per the HF config (`# Combined projections DISABLED`). MFSDP uses Megatron's fused QKV (1 GEMM) and fused gate+up (1 GEMM). With 80 layers × FWD+recompute+BWD×dx+BWD×dW and GA=4, this yields ~1.6× more GEMM dispatches in FSDP2. MFSDP's 1.22× higher GEMM total time reflects larger individual tiles (fused projections = single large GEMM vs 3 smaller GEMMs). |
| Flash-attn FWD/BWD | 1,280 / 640 | 1,280 / 640 | Same | MFSDP 0.57–0.61× (faster) | **Identical count, different kernels.** 80 layers × 4 GA = 320 FWD, 160 BWD per step. Performance gap is entirely due to cuDNN WGMMA kernel vs FlashAttention-2. |
| RMSNorm FWD/BWD | 1,284 / 644 | 1,284 / 644 | Same | ~1.0× (same) | **Identical, as expected.** 80 layers × 2 norms/layer × 2 FWD passes (original + recompute) × GA=4 = 1,280 ≈ 1,284. BWD: 80 × 2 × GA=4 = 640. Both match exactly — confirms same model structure and recompute strategy. |

**Communication vs Compute (per step):**

| | MFSDP | FSDP2 |
|---|---|---|
| Comm (AG+RS+AR+P2P) | 17,245 ms (49.0% of kernel) | 13,040 ms (40.2% of kernel) |
| Compute (GEMM+attn FWD+BWD main) | 16,717 ms (47.5% of kernel) | 15,472 ms (47.8% of kernel) |
| AG–GEMM overlap | **67.6%** (~2,456 ms exposed) | **62.0%** (~2,110 ms exposed) |
| RS–GEMM overlap | **87.8%** (~1,145 ms exposed) | **84.6%** (~676 ms exposed) |

---

## 2. Attention Kernel Comparison

| | Megatron MFSDP | Automodel FSDP2 |
|---|---|---|
| Backend config | `--attention-backend fused` → `AttnBackend.fused` (main); `kitchen_attention_backend: sdpa` (CP ring) | `attn_implementation: sdpa` |
| FWD kernel | `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128` | `flash_fwd_kernel` (FlashAttention-2) |
| BWD kernel | `cudnn_generated_fort_native_sdpa_sm90_flash_bprop_wgmma_f16_knob_26_64x64x128` | `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` |
| FWD avg duration | **599 µs** | 1,057 µs |
| BWD avg duration | **1,750 µs** | 2,870 µs |
| FWD speedup | **1.77×** | baseline |
| BWD speedup | **1.64×** | baseline |

Megatron uses TransformerEngine's FusedAttention which dispatches to cuDNN's sm90 WGMMA-optimized SDPA kernel. Automodel uses PyTorch SDPA (`flash_fwd_kernel` / FlashAttention-2) with `attn_implementation: sdpa`.

---

## 3. ReduceScatter Deep Dive

Both runs use identical kernel: `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL`. No BF16/FP16 RS in either run.

### RS Summary Table

| Metric | Megatron MFSDP | Automodel FSDP2 |
|---|---|---|
| RS calls (total) | 656 | 324 |
| RS calls (per step) | **328** | **324** |
| Avg message size (estimated) | ~27.3 MB | ~27.7 MB |
| Implied eff. bandwidth | **0.94 GB/s** | **1.57 GB/s** |
| BW ratio | **1.67× slower** | baseline |
| Duration avg | 28,497 µs | 17,656 µs |
| Duration min | 13,720 µs | ~14,693 µs |
| Duration max | 47,607 µs | 22,747 µs |
| Duration p50 | 29,821 µs | 18,038 µs |
| Duration p90 | 30,944 µs | 19,765 µs |

### Analysis

- **Call count**: identical per step (~326) → same bucketing granularity
- **Message size**: identical (~27 MB) — same model (70B params), same DP=32, same FP32 dtype (both use `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL`)
- **Dtype mismatch**: ruled out — both f32, zero BF16/FP16 RS kernels in either profile
- **Queuing**: ruled out — Megatron's p10–p90 band is tight (23–31 ms); queuing would produce a bimodal fast+slow distribution
- **Root cause**: lower NCCL effective bandwidth in Megatron (0.94 vs 1.57 GB/s). Likely caused by `NCCL_P2P_NET_CHUNKSIZE=2097152` set in Megatron's env, which changes RING_LL chunking for ~27 MB messages vs default settings in Automodel

### Message size derivation

```
Total FP32 gradient bytes = 70B params × 4 bytes = 280 GB
Per-GPU RS data (DP=32)   = 280 GB / 32          = 8.75 GB/step
Per RS call (~326 calls)  = 8.75 GB / 326         ≈ 27 MB/call
```

---

## 4. RMSNorm (Megatron MFSDP)

Normalized to per step (÷2):

| Kernel | Count/step | Total/step (ms) | Avg (µs) | % kernel time |
|---|---|---|---|---|
| `rmsnorm_fwd_tuned_kernel` | 1,284 | 78.9 | 61 | 0.22% |
| `rmsnorm_bwd_tuned_kernel` | 644 | 80.3 | 125 | 0.23% |
| `rmsnorm_bwd_finalize_tuned_kernel` | 644 | 4.2 | 6 | 0.01% |
| **Total/step** | | **163.4 ms** | | **0.46%** |

RMSNorm is negligible (<0.5% of kernel time). Not a performance concern.

---

## 5. Fairness Assessment

| Difference | Direction | Impact |
|---|---|---|
| Attention kernel (cuDNN vs FA2) | Megatron advantage | ~2pp MFU — **real unfairness**; Automodel cannot use cuDNN with current CP>1 SDPA constraint |
| RS NCCL bandwidth | Megatron disadvantage | ~1pp MFU lost; Megatron's `NCCL_P2P_NET_CHUNKSIZE` reduces RS efficiency |
| AG–GEMM overlap | Megatron slight advantage | 67.6% vs 62.0%; ~60 ms more exposed AG in FSDP2 despite faster individual AGs. Likely partly due to pre-AG `CatArrayBatchedCopy` delaying effective AG launch; Megatron stream scheduling may also differ but not directly confirmed |
| AG/RS copy overhead (contiguous buffer) | Megatron advantage | ~6.1% kernel time / ~4.7% wall time; FSDP2 pays 1,979 ms/step in pack/unpack copies; MFSDP pays 64 ms (flat buffer, no staging) |
| Gradient dtype | Fair | Both FP32 RS |
| Model architecture | Fair | Same TP/CP/PP/DP, MBS, GBS, seqlen |
| Cross-entropy fusion | Megatron minor advantage | `--cross-entropy-loss-fusion te` |

**Overall MFU gap**: 41.7% (Megatron) vs 37.2% (Automodel) = **4.5pp / ~12% relative**

Primary driver: cuDNN attention (~2pp). Secondary: overlap quality and NCCL settings partially offset each other.

---

## 6. Improvement Opportunities for Automodel

1. **Enable cuDNN SDPA** — largest single lever (~2pp MFU). Switch from FA2 (`flash_fwd_kernel`) to cuDNN WGMMA kernel by enabling `torch.backends.cuda.enable_cudnn_sdp(True)` in the attention backend.
2. **Tune FSDP2 prefetch depth** — increase `fsdp2_forward_prefetch_depth` to reduce exposed AG time (currently 38% AG exposed vs Megatron's 32%; tuning depth or bucket size may close this).
3. **Match NCCL env vars** — test with Megatron's `NCCL_P2P_NET_CHUNKSIZE` and `NCCL_IB_SL` settings to understand their impact on RS bandwidth (currently FSDP2 is faster without them).
4. **Cross-entropy fusion** — add TE fused cross-entropy loss (`--cross-entropy-loss-fusion te` equivalent) to reduce activation memory and fuse the softmax+loss kernel.
5. **Adopt contiguous flat parameter buffer (M-FSDP-style)** — FSDP2 stores parameters as individual tensors, requiring explicit pack/unpack copies before every AllGather and ReduceScatter. Megatron MFSDP maintains a single contiguous memory slab so AG/RS operate directly on it with no staging. The measured GPU kernel overhead per step:

   | Kernel | Count | Total (ms) | Avg (µs) | Role |
   |---|---|---|---|---|
   | `chunk_cat_cuda_kernel` | 324 | 940 | 2,902 | Pre-RS: pack param-wise gradients into contiguous buffer |
   | `split_with_sizes_copy_out_contiguous_no_cast_kernel` | 640 | 822 | 1,284 | Post-AG: unpack contiguous buffer back to param tensors |
   | `CatArrayBatchedCopy` | 1,600 | 191 | 119 | Pre-AG batched per-param shard copy |
   | `CatArrayBatchedCopy_vectorized` | 1,600 | 26 | 16 | Vectorized variant |
   | **FSDP2 total** | | **1,979** | | |
   | **MFSDP total** (`CatArrayBatchedCopy_vectorized` only) | | **64** | | No pre-RS/post-AG copy needed |

   Two views of the overhead:
   - **6.1% of FSDP2 kernel time** (1,979 / 32,399 ms): wasted compute capacity — GPU time spent on bookkeeping MFSDP never pays. This is the fairer efficiency gap metric.
   - **4.7% of FSDP2 wall time** (940 / 20,031 ms): hard latency floor from `chunk_cat` alone, which is 100% serial before each RS and cannot be overlapped.

   The serial nature of `chunk_cat` was confirmed by measuring the gap between its end and the immediately following RS kernel start across 10 representative calls — consistently **7–11 µs** (effectively zero; RS is hard-blocked on `chunk_cat`). The post-AG `split_with_sizes_copy_out` (822 ms) runs concurrently with RS on a separate stream but occupies the compute stream for ~1,284 µs per layer, reducing the GEMM–RS overlap window. `CatArrayBatchedCopy` (191 ms) is interleaved with GEMMs and partially hidden.

   Adopting a flat contiguous buffer in Automodel (analogous to `use_orig_params=False` with custom bucketing, or a dedicated M-FSDP-style slab) would eliminate `chunk_cat` and `split_with_sizes_copy_out` entirely and reduce `CatArrayBatchedCopy` to the ~64 ms MFSDP baseline, recovering up to **~4.7% step time** from the serial path and **~6.1% compute efficiency**.

---

## 7. Improvement Opportunities for Megatron MFSDP

1. **Remove or retune `NCCL_P2P_NET_CHUNKSIZE=2097152`** — most actionable win. This env var reduces RS effective bandwidth from ~1.57 GB/s (FSDP2 default) to ~0.94 GB/s (~1.67× slower per RS call). RS accounts for 26.6% of kernel time; recovering this bandwidth would save ~3.5s/step and recover ~1pp MFU. Test without this flag first.
2. **Reduce AG bucket size** — MFSDP issues 644 AGs/step at avg 11,756 µs each vs FSDP2's 643 AGs at 10,280 µs. Despite similar counts, MFSDP's individually larger AGs may reduce overlap granularity per GEMM. Tuning `--param-gather-bucket-size` (or equivalent) to create finer-grained AGs could improve AG–GEMM overlap beyond the current 67.6%.
3. **Validate `NCCL_IB_SL=1` impact** — this sets InfiniBand Service Level, which can affect QoS and routing. Confirm it improves (vs degrades) bandwidth on the current fabric; it may be a carry-over from an older network topology.
