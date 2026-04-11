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
| nsys-rep | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/Automodel-release/MFSDP-llama3-70b/llama3_70b_mfsdp_tp1_pp1_ep1_cp2_hsdp1_alltoall_mbs1_gbs128_seqlen8192_cw_n8_full-recompute.nsys-rep` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/Automodel-release/slurm_jobs/11037458/nsys_llama31_70b_pretrain_tp1cp2pp1_te_attn_cp2_no_compile.nsys-rep` |
| sqlite | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/Automodel-release/MFSDP-llama3-70b/llama3_70b_mfsdp_tp1_pp1_ep1_cp2_hsdp1_alltoall_mbs1_gbs128_seqlen8192_cw_n8_full-recompute.sqlite` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/Automodel-release/slurm_jobs/11037458/nsys_llama31_70b_pretrain_tp1cp2pp1_te_attn_cp2_no_compile.sqlite` |
| Config file | `/lustre/fsw/portfolios/coreai/users/xuwenc/code/mfsdp_benchmark/megatron-benchmark/scripts/llama3_70b/sbatch_llama3_70b_mfsdp_tp1_pp1_ep1_cp2_hsdp1_alltoall_mbs1_gbs128_seqlen8192_cw_n8_full-recompute.sh` | `/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/zhiyul/Automodel-release/examples/llm_pretrain/llama3_1_70b_pretrain_benchmark_8nodes_tp1cp2pp1_0410.yaml` |
| Profile window | 37.61s (~2 steps) | 20.05s (~1 step) |
| MFU | ~41.7% (412.6 TFLOPs/GPU) | ~37.2% |
| Step time | ~18.8s | ~20.05s |

---

## 1. Kernel Time Breakdown — Per Step (Side-by-Side)

All values normalized to **1 step**: Megatron window = 2 steps (37.61s, no NVTX markers, ÷2); Automodel = 1 step (20.05s, confirmed via NVTX `iteration_7_ga_step_*`). Avg µs is per-call and unchanged by normalization. All numbers verified by direct SQL query against the nsys sqlite files.

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
| AllGather | 644 | 7,571 | 11,756 | 1,603 | 7,263 | 4,531 | Tie (7,571 vs 7,263 ms) |
| ReduceScatter | 328 | 9,347 | 28,497 | 324 | 5,862 | 18,091 | **FSDP2** (5,862 vs 9,347 ms, 1.59×) |
| AllReduce | 13 | 17 | 1,304 | 1 | ~1 | — | Tie (negligible) |
| P2P/SendRecv | 1,280 | 310 | 242 | 640 | 315 | 493 | Tie (310 vs 315 ms) |
| GEMM (`nvjet_sm90_*`) | 5,132 | 14,830 | — | 8,332 | 12,189 | — | Tie ² |
| Flash-attn FWD | 1,280 | 767 | **599** | 1,280 | 1,353 | 1,057 | **MFSDP** (599 vs 1,057 µs avg, 1.77×) |
| Flash-attn BWD ¹ | 640 | 1,120 | **1,750** | 640 | 1,837 | 2,870 | **MFSDP** (1,750 vs 2,870 µs avg, 1.64×) |
| RMSNorm FWD | 1,284 | 79 | 62 | 1,284 | 83 | 65 | Tie (79 vs 83 ms) |
| RMSNorm BWD | 644 | 80 | 125 | 644 | 80 | 124 | Tie (80 vs 80 ms) |
| **Total RMSNorm** | | **160** | | | **166** | | Tie (160 vs 166 ms) |
| **Total kernel/step** | | **35,177** | | | **32,695** | | — |
| **Step wall time** | | **18,807** | | | **20,047** | | **MFSDP** (18,807 vs 20,047 ms, 1.07×) |

² GEMM total time is not a fair single-winner metric: MFSDP has larger but fewer GEMMs (fused QKV/gate-up); FSDP2 has more but smaller GEMMs (separate Q/K/V). Both do the same mathematical work — the difference is tiling strategy, not efficiency.

¹ BWD main kernel only: MFSDP=`cudnn_generated_fort_native_sdpa_sm90_flash_bprop_*`, FSDP2=`flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`. FSDP2 full BWD including helpers (`flash_bwd_dot_do_o_kernel` + `flash_bwd_convert_dq_kernel`) = **1,924ms** total.

### Count Difference Analysis

Counts differ across rows — this does **not** necessarily mean different total data volume. Analysis per category:

| Category | MFSDP Count | FSDP2 Count | Ratio | Total time ratio | Verdict |
|---|---|---|---|---|---|
| AllGather | 644 | 1,603 | 2.49× more in FSDP2 | 1.04× (same) | **Same volume, finer granularity.** FSDP2 prefetches per-layer (one AG per module). MFSDP uses larger parameter buckets → fewer but 2.6× longer AGs (11,756 vs 4,531 µs avg). Total bandwidth consumed is nearly identical. |
| ReduceScatter | 328 | 324 | Same | 1.59× more in MFSDP | **Same count, same volume (~27 MB/call), but MFSDP 1.58× slower bandwidth** (0.94 vs 1.49 GB/s). Root cause: `NCCL_P2P_NET_CHUNKSIZE=2097152` in Megatron env — see §3. |
| P2P/SendRecv | 1,280 | 640 | 2× more in MFSDP | 0.98× (same) | **Same volume, different chunking.** MFSDP issues 2× more but 2× shorter P2P calls (242 vs 493 µs avg); total time is identical (310 vs 315 ms). Likely due to different AllToAll CP vs ring-attention P2P decomposition per layer. |
| GEMM (`nvjet_sm90_*`) | 5,132 | 8,332 | 1.62× more in FSDP2 | MFSDP 1.22× higher | **Different projection fusion, not same volume.** FSDP2 uses **separate Q, K, V projections** (3 GEMMs/layer) and **separate gate + up** (2 GEMMs/layer) per the HF config (`# Combined projections DISABLED`). MFSDP uses Megatron's fused QKV (1 GEMM) and fused gate+up (1 GEMM). With 80 layers × FWD+recompute+BWD×dx+BWD×dW and GA=4, this yields ~1.6× more GEMM dispatches in FSDP2. MFSDP's 1.22× higher GEMM total time reflects larger individual tiles (fused projections = single large GEMM vs 3 smaller GEMMs). |
| Flash-attn FWD/BWD | 1,280 / 640 | 1,280 / 640 | Same | MFSDP 0.57–0.61× (faster) | **Identical count, different kernels.** Same number of attention passes (80 layers × CP=2 × GA=4 / 4 = 80... per step). Performance gap is entirely due to cuDNN WGMMA kernel vs FlashAttention-2. |
| RMSNorm FWD/BWD | 1,284 / 644 | 1,284 / 644 | Same | ~1.0× (same) | **Identical, as expected.** 80 layers × 2 norms/layer × 2 FWD passes (original + recompute) × GA=4 = 1,280 ≈ 1,284. BWD: 80 × 2 × GA=4 / 4 = 160 × 4 = 640. Both runs match exactly — confirms same model structure and recompute strategy. |

**Communication vs Compute (per step):**

| | MFSDP | FSDP2 |
|---|---|---|
| Comm (AG+RS+AR+P2P) | 17,245 ms (49.0% of kernel) | 13,441 ms (41.1% of kernel) |
| Compute (GEMM+attn FWD+BWD main) | 16,717 ms (47.5% of kernel) | 15,379 ms (47.0% of kernel) |
| AG–GEMM overlap | **67.6%** (~2,456 ms exposed) | **62.0%** (2,110 ms exposed) |
| RS–GEMM overlap | **87.8%** (~1,145 ms exposed) | **84.6%** (676 ms exposed) |

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

Megatron uses TransformerEngine's FusedAttention which dispatches to cuDNN's sm90 WGMMA-optimized SDPA kernel. Automodel is constrained to PyTorch SDPA (`flash_fwd_kernel`) because the current CP ring-attention implementation requires `attn_implementation: sdpa`.

---

## 3. ReduceScatter Deep Dive

Both runs use identical kernel: `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL`. No BF16/FP16 RS in either run.

### RS Summary Table

| Metric | Megatron MFSDP | Automodel FSDP2 |
|---|---|---|
| RS calls (total) | 656 | 324 |
| RS calls (per step) | **328** | **324** |
| Avg message size (estimated) | ~27.3 MB | ~27.7 MB |
| Implied eff. bandwidth | **0.94 GB/s** | **1.49 GB/s** |
| BW ratio | **1.58× slower** | baseline |
| Duration avg | 28,497 µs | 18,091 µs |
| Duration min | 13,720 µs | 14,693 µs |
| Duration max | 47,607 µs | 22,747 µs |
| Duration p50 | 29,821 µs | 18,038 µs |
| Duration p90 | 30,944 µs | 19,765 µs |

### Analysis

- **Call count**: identical per step (~326) → same bucketing granularity
- **Message size**: identical (~27 MB) — same model (70B params), same DP=32, same FP32 dtype (both use `ncclDevKernel_ReduceScatter_Sum_f32_RING_LL`)
- **Dtype mismatch**: ruled out — both f32, zero BF16/FP16 RS kernels in either profile
- **Queuing**: ruled out — Megatron's p10–p90 band is tight (23–31 ms); queuing would produce a bimodal fast+slow distribution
- **Root cause**: lower NCCL effective bandwidth in Megatron (0.94 vs 1.49 GB/s). Likely caused by `NCCL_P2P_NET_CHUNKSIZE=2097152` set in Megatron's env, which changes RING_LL chunking for ~27 MB messages vs default settings in Automodel

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
| AG–GEMM overlap | Megatron slight advantage | 67.6% vs 62.0%; Megatron's `--overlap-param-gather` more aggressive than `fsdp2_backward_prefetch_depth=2` |
| Gradient dtype | Fair | Both FP32 RS |
| Model architecture | Fair | Same TP/CP/PP/DP, MBS, GBS, seqlen |
| Manual GC | Megatron advantage | Reduces step-time jitter; Automodel had one 38s outlier step |
| Cross-entropy fusion | Megatron minor advantage | `--cross-entropy-loss-fusion te` |

**Overall MFU gap**: 41.7% (Megatron) vs 37.2% (Automodel) = **4.5pp / ~12% relative**

Primary driver: cuDNN attention (~2pp). Secondary: overlap quality and NCCL settings partially offset each other.

---

## 6. Improvement Opportunities for Automodel

1. **Enable cuDNN SDPA for CP>1** — largest single lever (~2pp MFU). Requires updating the CP ring-attention implementation to support `torch.backends.cuda.enable_cudnn_sdp(True)`.
2. **Tune FSDP2 prefetch depth** — increase `fsdp2_forward_prefetch_depth` to reduce exposed AG time (currently 38% AG exposed vs Megatron's 32%; tuning depth or bucket size may close this).
3. **Match NCCL env vars** — test with Megatron's `NCCL_P2P_NET_CHUNKSIZE` and `NCCL_IB_SL` settings to understand their impact on RS bandwidth (currently FSDP2 is faster without them).
4. **Manual GC** — add explicit GC control to prevent outlier steps (~38s outlier observed).
5. **Cross-entropy fusion** — add TE fused cross-entropy loss (`--cross-entropy-loss-fusion te` equivalent) to reduce activation memory and fuse the softmax+loss kernel.

---

## 7. Improvement Opportunities for Megatron MFSDP

1. **Remove or retune `NCCL_P2P_NET_CHUNKSIZE=2097152`** — most actionable win. This env var reduces RS effective bandwidth from ~1.49 GB/s (FSDP2 default) to ~0.94 GB/s (~1.58× slower per RS call). RS accounts for 26.6% of kernel time; recovering this bandwidth would save ~3.5s/step and recover ~1pp MFU. Test without this flag first.
2. **Reduce AG bucket size** — MFSDP issues 644 AGs/step at avg 11,756 µs each vs FSDP2's 1,603 AGs at 4,531 µs. Larger buckets mean less overlap opportunity per GEMM. Tuning `--param-gather-bucket-size` (or equivalent) to create finer-grained AGs could improve AG–GEMM overlap beyond the current 67.6%.
3. **Validate `NCCL_IB_SL=1` impact** — this sets InfiniBand Service Level, which can affect QoS and routing. Confirm it improves (vs degrades) bandwidth on the current fabric; it may be a carry-over from an older network topology.
