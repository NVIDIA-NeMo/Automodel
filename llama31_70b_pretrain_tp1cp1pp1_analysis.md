# LLaMA 3.1 70B Pretrain — tp1cp1pp1 Overlap Analysis

**Profile**: `slurm_jobs/11082936/nsys_llama31_70b_pretrain_tp1cp1pp1_te_attn_cp1_no_compile_node0.nsys-rep`  
**Config**: TP=1, CP=1, PP=1, TransformerEngine attention (cuDNN SDPA), no torch.compile  
**Hardware**: 1 node × 8 × H100 (NVLink), FSDP2 over 8 GPUs  
**Analysis tool**: custom per-device overlap script (see below)

---

## Steady-State Window

| Metric | Value |
|--------|-------|
| Profiled iterations | iteration 7, ga_steps 1–3 (step 0 is warmup/compile, excluded) |
| Steps analyzed | 3 |
| Step wall time (avg) | **4312 ms** |
| Total window | 12.937 s |

---

## Stream Layout (Device 0 / Rank 0)

> Device 0 uses different stream IDs than devices 1–7. All 8 devices are present in this single `.sqlite` file. Analysis was validated on both device 0 and device 1.

| Stream | Role | Total GPU time | Avg kernel dur |
|--------|------|---------------|----------------|
| 7 | Main compute (GEMMs, attention, ops) | 12.38 s | — |
| 24 | Secondary compute (small ops) | 0.055 s | — |
| 27 | Small compute (oneRankReduce, misc) | 0.160 s | — |
| **28** | **FSDP2 AllGather** | **4.802 s** | 10.0 ms |
| **32** | **FSDP2 ReduceScatter** | **5.174 s** | 21.3 ms |

Devices 1–7 use stream 21 (compute), stream 42 (AllGather), stream 46 (ReduceScatter) — all showing consistent timing within ±3%.

---

## AllGather / ReduceScatter Overlap Analysis

Overlap is measured by computing the **intersection** of merged NCCL kernel intervals with merged compute kernel intervals (all compute streams: 7+24+27 for device 0), restricted to the steady-state window.

The key distinction: **exposed** = NCCL GPU time with **no concurrent compute on any other stream** — this is the stall cost.

### Device 0 Results

#### FSDP2 AllGather (stream 28)

| Metric | Total (3 steps) | Per step |
|--------|----------------|---------|
| AG kernels | 480 | 160 |
| AG GPU time (raw sum) | 4.802 s | 1600.6 ms |
| Overlapped with compute | 4.696 s (97.8%) | 1565.4 ms |
| **Exposed (no compute)** | **0.105 s (2.2%)** | **35.2 ms** |
| Exposed as % of wall | — | **0.8%** |

#### FSDP2 ReduceScatter (stream 32)

| Metric | Total (3 steps) | Per step |
|--------|----------------|---------|
| RS kernels | 243 | 81 |
| RS GPU time (raw sum) | 5.174 s | 1724.6 ms |
| Overlapped with compute | 4.929 s (95.3%) | 1643.2 ms |
| **Exposed (no compute)** | **0.244 s (4.7%)** | **81.4 ms** |
| Exposed as % of wall | — | **1.9%** |

#### Summary

| | GPU time/step | Overlap | Exposed/step | % of wall |
|---|---|---|---|---|
| AllGather | 1600.6 ms | 97.8% | 35.2 ms | 0.8% |
| ReduceScatter | 1724.6 ms | 95.3% | 81.4 ms | 1.9% |
| **Total NCCL** | **3325.2 ms** | **96.5%** | **116.6 ms** | **2.7%** |

**Device 1 cross-check** (streams 21/42/46): AG exposed 37.2ms/step, RS exposed 82.6ms/step, total 119.7ms/step (2.8% of wall) — consistent with device 0 within 3ms.

AG and RS have **zero concurrent overlap** with each other (sequential in all steady-state steps).

---

## Compute Breakdown (Device 0, Per Step)

| Kernel type | GPU time/step | Count/step | Notes |
|-------------|--------------|------------|-------|
| GEMM (nvjet_sm90 TNT) | 950.8 ms | 321× | Largest GEMM variant |
| GEMM (nvjet_sm90 NNT) | 648.3 ms | 481× | |
| GEMM (nvjet_sm90 NTN) | 483.9 ms | 161× | |
| GEMM (nvjet_sm90 TNT-small) | 456.6 ms | 640× | |
| Other GEMMs | 647.1 ms | — | NNT/NTT/NTN variants |
| **Total GEMM** | **3186.5 ms** | — | 73.9% of wall |
| cuDNN SDPA fwd | 92.4 ms | 160× | TE attention, replaces flash_fwd |
| cuDNN SDPA bwd | 134.1 ms | 80× | TE attention, replaces flash_bwd |
| **Total TE attention** | **226.5 ms** | — | 5.3% of wall |
| chunk_cat + split_with_sizes | 440.2 ms | — | FSDP2 tensor reshape/concat |
| Elementwise ops | 196.3 ms | 1932× | |
| RMSNorm fwd+bwd | 41.7 ms | — | |
| RoPE fwd+bwd | 24.7 ms | 320× | |
| Other (reduce, misc) | 15.3 ms | — | |
| **Total compute stream** | **4131.2 ms** | — | **95.8% of wall** |

---

## Implementation Review of Automodel Overlap Scripts

### `analyze_overlap.py` (sweep-line, AllGather vs GEMM)

**Algorithm**: Correct. Two-pointer sweep-line in O((N_ag + N_gemm) log N) is efficient and the union-of-GEMM-intervals logic is correct. The left_ptr advance is valid since AllGather kernels are sorted by start.

**Limitation — GEMM-only comparison**: Compares AllGather only against `%nvjet_sm90%` (cutlass GEMMs), missing cuDNN attention (226ms/step), elementwise, RMSNorm, etc. This causes it to **overestimate exposed AG time** by up to ~226ms/step in this config (where TE attention runs concurrently with AllGather).

**Limitation — no ReduceScatter analysis**: RS overlap is not computed. In this profile RS is the larger exposed cost (81ms vs 35ms).

**Correct**: device filtering via `deviceId = ?`. Steady-state bounds via NVTX `iteration_%_ga_step_%` excluding `_ga_step_0`.

### `analyze_allgather_overlap.py` (per-kernel, multi-stream compute)

**Algorithm**: Correct but O(N_ag × N_streams × N_intervals_per_stream) — acceptable for smaller profiles but slower.

**Strength**: Uses a broader set of compute patterns (GEMM, flash, elementwise, triton, softmax, etc.), giving more accurate overlap than `analyze_overlap.py`.

**Note**: The `break` in `covered_by_other_streams` is correct — intervals within each stream are sorted and merged, so once `ms >= ag_e` no further intervals can overlap.

**Limitation**: Backward-only (uses `bwd_start` from first `flash_bwd` kernel). In TE attention profiles, backward is identified by `%flash_bwd%` — **this will not find the backward start** since TE uses `cudnn_generated_fort_native_sdpa_sm90_flash_bprop_*`. The `bwd_start` query will return `None`, causing the script to crash or analyze the wrong window.

### `analyze_tp1cp2pp1_overlap.py` (stream-based, correct per-stream intervals)

**Algorithm**: Correct. Merge intervals per NCCL type, compute intersection with compute streams. Handles cross-stream AG overlap correctly.

**Limitation**: Stream IDs (7, 28, 32, 42, 46) are **hardcoded for a specific profile** (tp1cp2pp1 with CP). For tp1cp1pp1 (this profile), device 0 uses streams 28/32 for FSDP2 and stream 7 for compute — which accidentally matches the hardcoded values. But device 1–7 use streams 42/46 for FSDP2, which would be misclassified as CP streams.

**No deviceId filter**: queries without `WHERE deviceId = ?` aggregate across all 8 GPUs, inflating all times by 8×.

---

## Recommendations for Future Analysis

1. **Always filter by `deviceId`** when the SQLite contains multiple ranks (e.g., 8 GPUs). Use device 0 for consistency with NVTX annotations.

2. **Discover streams dynamically** rather than hardcoding. Query `GROUP BY streamId` with NCCL kernel patterns to find the actual NCCL streams for the device being analyzed.

3. **Include cuDNN attention in compute patterns**. For TE attention profiles, add `%cudnn_generated_fort%` or `%sdpa%` to the compute patterns.

4. **Analyze ReduceScatter separately** — it is often a larger exposed cost than AllGather in FSDP2 backward pass.

5. **Use the bwd_start from NVTX** (e.g., `bwd_loss_scale` or `iteration_%` NVTX range) rather than `flash_bwd` kernel for TE profiles.

---

## Roofline and Exposure Breakdown

### AllGather — 35.2 ms/step exposed (2.2% of AG GPU time = 0.8% of wall)

| Category | Kernels | ms/step | % of AG exposure | Root cause |
|----------|---------|---------|-----------------|------------|
| ① Fully exposed | 3 (1/step) | **16.6** | **47%** | First AG can't prefetch across step boundary — **ROOFLINE** |
| ② Partial — startup gap | 3 (1/step) | **5.4** | **15%** | Second AG fires before compute resumes; avg gap = 689 µs |
| ③ Partial — tail gap | 0 | 0.0 | 0% | n/a |
| ④ Partial — inner jitter | 474 (~158/step) | **13.2** | **38%** | ~83 µs gap per layer × 160 AG/step (kernel launch latency) |
| **Total** | | **35.2** | 100% | |
| **Roofline (①)** | | **16.6** | | |
| **Avoidable (②④)** | | **18.6** | | |

### ReduceScatter — 81.4 ms/step exposed (4.7% of RS GPU time = 1.9% of wall)

| Category | Kernels | ms/step | % of RS exposure | Root cause |
|----------|---------|---------|-----------------|------------|
| ① Fully exposed | 0 | 0.0 | 0% | — (tiny stream-27 ops always present) |
| ② Partial — startup gap | 3 (1/step) | **4.7** | **6%** | RS fires ~4.4 ms into a compute gap (first backward RS) |
| ③ Partial — tail gap | 8 (~2.7/step) | **70.1** | **86%** | Compute finishes before RS ends; avg tail = 26 ms — **PRACTICAL ROOFLINE** |
| ④ Partial — inner jitter | 232 (~77/step) | **6.7** | **8%** | ~71 µs gap per layer × 81 RS/step |
| **Total** | | **81.4** | 100% | |
| **Practical roofline (③)** | | **70.1** | | last 2-3 RS at bwd end |
| **Avoidable (②④)** | | **11.3** | | |

### Summary table

| | AG | RS | Total |
|---|---|---|---|
| GPU time/step | 1601 ms | 1725 ms | 3325 ms |
| **Exposed/step** | **35.2 ms** | **81.4 ms** | **116.6 ms** |
| Exposed % of wall | 0.8% | 1.9% | 2.7% |
| — ① Roofline | 16.6 ms (0.4%) | ~70 ms (1.6%) | ~87 ms (2.0%) |
| — Avoidable overhead | 18.6 ms (0.4%) | 11.3 ms (0.3%) | ~30 ms (0.7%) |

---

### What each category means

**① Fully exposed — ROOFLINE**
- **AG**: The first AllGather of every step fetches layer-0 weights. FSDP2 cannot prefetch across step boundaries, so this AG runs with zero concurrent compute. With 160 AG/step and avg 10 ms each, the theoretical minimum exposed is `1/160 × 1600 ms = 10 ms/step`. Actual is 16.6 ms because step-3's first AG is 23 ms (network jitter vs 12–14 ms in steps 1–2).
- **RS**: Technically zero because stream-27 (oneRankReduce, <0.1 ms) always runs a handful of tiny kernels. The practical equivalent is category ③ below.

**② Partial — startup gap**
- **AG**: Three kernels (one per step) = the *second* AG at step start. Right after the first AG completes, the second AG fires immediately on stream 28, but the unshard pipeline hasn't handed off to stream 7 yet. Avg gap = 689 µs before compute resumes. Exposed ≈ 5.4 ms/step.
- **RS**: Three kernels, avg 4.4 ms startup gap. These are RS kernels that land in a compute-quiet window (likely the very first backward RS, before stream-7 backward compute is fully pipelined in).

**③ Partial — tail gap (end-of-backward RS) — PRACTICAL RS ROOFLINE**
- Eight RS kernels total (~2.7 per step), avg tail gap = **26 ms**. These are the last 2–3 RS operations of each backward pass: after stream-7 backward compute finishes, these RS kernels are still running with no subsequent layer to overlap with.
- Main contributors per step: the last-layer RS (22–30 ms fully exposed tail) plus the **embedding-table RS** (~63 ms on step 3, 3× the average RS duration because the embedding is 128 256 × 8 192 params across 64 GPUs over inter-node IB).
- This is structurally unavoidable unless (a) RS is made faster, (b) the backward compute after the last RS is extended (e.g., extra fused ops), or (c) async optimizer overlap is implemented.

**④ Partial — inner jitter (launch latency)**
- ~474 AG kernels and ~232 RS kernels each have small compute gaps mid-flight: avg **83 µs for AG**, **71 µs for RS**. This is CUDA kernel launch latency between adjacent ops on stream 7. When layer `i`'s last compute kernel completes and layer `i+1`'s first compute kernel is dispatched, there is a ~80–100 µs gap during which the overlapping NCCL kernel sees no compute. Multiplied by 160 AG/step → 13 ms/step; by 81 RS/step → 6.7 ms/step. Reducible with `torch.compile` or tighter kernel fusion.

---

## Roofline Analysis (condensed)

### AllGather Roofline

LLaMA 3.1 70B has 80 transformer layers. With FSDP2 and one forward + one backward pass per step:
- 80 AG in forward (one per layer) + 80 AG in backward = **160 AG/step**
- Average AG duration: **10.0 ms**
- Total FSDP2 AG GPU time: **1600 ms/step**

**Theoretical minimum (roofline):** FSDP2 prefetch can overlap layer `i+1`'s AG with layer `i`'s compute — but layer 0 (the very first layer of each step's forward pass) **cannot be prefetched across step boundaries**. This is an unavoidable 1/160 exposure:

```
Roofline AG exposed = 1 × avg_AG_dur = 10 ms/step
                    = 1/160 × 1600 ms/step
                    = 1/80 × 800 ms (forward-only framing)
```

**Actual: 35.2 ms/step → 3.5× roofline**

### ReduceScatter Roofline

- 81 RS/step (80 layers in backward + 1 embedding/final RS)
- Average RS duration: **21.3 ms**

**Theoretical minimum:** The last RS of backward (layer 0's gradient) has no subsequent compute to hide behind:

```
Roofline RS exposed ≈ 1 × avg_RS_dur = 21 ms/step
```

**Actual: 81.4 ms/step → 3.8× roofline**

---

## Exposed Communication Breakdown

### AllGather: 35.2 ms/step (105.5 ms over 3 steps)

| Category | Kernels | Exposed time | % of total AG exposure |
|----------|---------|-------------|----------------------|
| **Fully/near-fully exposed (>50%)** | 6 | 65.9 ms | **62%** |
| Medium partial (100–500 µs) | 109 | 14.6 ms | 14% |
| Small partial (10–100 µs) | 364 | 25.0 ms | 24% |
| Negligible (<10 µs) | 1 | 0.01 ms | ~0% |

#### Root cause 1 — First AG of each step (62% of AG exposure)

The 6 fully-exposed kernels are **2 AG per step × 3 steps = the first two layers of each step's forward pass**:

```
Step start → [6 ms Python/CPU overhead: loss scale, input prep]
           → multi_tensor_apply (optimizer fragment, stream 24, 4×0.04 ms)
  +6.42 ms → AG layer 0 fires (stream 28), dur≈12–23 ms  ← FULLY EXPOSED
  +6.42 ms → AG layer 1 fires immediately after (stream 28), dur≈5.3–5.5 ms  ← FULLY EXPOSED
 +19.04 ms → split_with_sizes (unshard layer 0) → first real compute starts
```

- The first AG cannot be prefetched across step boundaries
- The second AG starts before compute kicks in (still in the unshard pipeline)
- Per step: ~18–29 ms from these first two layers (steps 1–3 vary: 17.9 / 19.9 / 28.2 ms)
- Step 3 is significantly worse (22.7 ms first AG vs 12.6–14.4 ms for steps 1–2) — likely inter-node network congestion on the last step

#### Root cause 2 — Systematic partial leakage (~38% of AG exposure)

473 AG kernels each contribute ~0.08–0.2 ms of exposure (≈2% of their 10 ms duration). This is **launch/scheduling jitter** at layer boundaries: compute on stream 7 has a ~100–200 µs tail gap before the next layer's kernels launch, during which the next AG has already started. Each AG "leaks" slightly before compute fills in. Total: ~39.6 ms / 3 steps = 13 ms/step.

---

### ReduceScatter: 81.4 ms/step (244.3 ms over 3 steps)

| Category | Kernels | Exposed time | % of total RS exposure |
|----------|---------|-------------|----------------------|
| **Mostly/fully exposed (>50%)** | 10 | 222.5 ms | **91%** |
| Partial (100 µs – 2 ms) | 13 | 4.9 ms | 2% |
| Small partial (10–100 µs) | 220 | 16.9 ms | 7% |

#### Root cause 1 — Last RS(es) of backward: no following compute (91% of RS exposure)

At the end of each backward pass, compute finishes before the last few RS operations complete. The last 3 RS kernels of each step are substantially exposed:

| Step | 3rd-to-last RS | 2nd-to-last RS | **Last RS** | Comment |
|------|---------------|---------------|-------------|---------|
| 1 | 18.8 ms | 16.9 ms (93% exp) | **29.8 ms (100%)** | large layer |
| 2 | 32.2 ms (39% exp) | 29.2 ms (96%) | **22.0 ms (100%)** | |
| 3 | 31.7 ms | 26.3 ms (96%) | **63.7 ms (100%)** | **embedding RS** |

The step 3 last RS is **63.7 ms** — 3× the average. This is the **embedding table gradient RS** (128 256 × 8 192 = ~1B params). Its RS involves all 64 GPUs (8 nodes × 8 GPUs) across inter-node IB, making it much more expensive than the transformer-layer RS operations.

The `chunk_cat_cuda_kernel` on stream 7 completes at `t=12.872 s`, immediately triggering the 63 ms embedding RS on stream 32. No compute follows → fully exposed.

#### Root cause 2 — Partial leakage: RS duration > layer compute duration (~9%)

220 RS kernels have small partial exposure (10–100 µs each). The transformer layer backward compute time (~10–15 ms) is occasionally shorter than the RS duration (~21 ms), leaving a small exposed tail. Total: ~17 ms / 3 steps = ~6 ms/step.

---

## Summary: Roofline vs Actual

| | Roofline | Actual | Overhead factor | Primary cause |
|---|---|---|---|---|
| AllGather | 10 ms/step | 35.2 ms/step | **3.5×** | First 2 layers can't prefetch at step start; 6ms Python startup delay |
| ReduceScatter | 21 ms/step | 81.4 ms/step | **3.8×** | Embedding RS is 3× avg (63ms); last 2-3 backward layers exposed |
| **Total NCCL** | **31 ms/step** | **116.6 ms/step** | **3.8×** | |

**Gap to roofline: ~86 ms/step** (2.0% of wall time)

Key bottlenecks in order of impact:
1. Embedding layer RS at end of backward: 63ms fully exposed (step 3) / ~30ms (steps 1–2)
2. First AG at step start (Python startup gap + no cross-step prefetch): ~12–23ms
3. Second layer AG at step start (still in unshard pipeline): ~5ms
4. Systematic 2% AG leakage per kernel × 160 kernels: ~13ms

---

## MFU Decomposition: Wall Time Breakdown

**Wall time is additive in two tiers:**
- **Compute-active tier** (4131 ms): time the compute stream is running kernels. All NCCL during this window is hidden/free.
- **Compute-idle tier** (181 ms): time the compute stream is idle, split between waiting for NCCL to drain (⑤) and pure gaps (⑥).

```
wall = compute_active + compute_idle
     = (GEMM + attention + light + copy) + (exposed_NCCL + idle_gaps)
     = 4131 + (116 + 65)
     = 4312 ms  ✓

NCCL (3325 ms/step) runs concurrently during compute_active → costs nothing extra.
Only the 116ms that spills past the end of compute shows up in wall time.
```

### Wall Time Decomposition (step = 4312 ms)

| Bucket | ms/step | % wall | Notes |
|--------|---------|--------|-------|
| **Compute-active (stream 7 busy)** | **4131** | **95.8%** | |
| ① GEMM (nvjet_sm90 cutlass) | 3186.5 | 73.9% | Linear projections, FFN |
| ② TE attention FWD+BWD (cuDNN SDPA) | 226.5 | 5.3% |  |
| ③ Light compute (norm/RoPE/elemwise/reduce) | 266.8 | 6.2% | RMSNorm 42ms + RoPE 25ms + elemwise 196ms + misc 4ms |
| ④ FSDP2 copy (chunk_cat + split_with_sizes) | 440.3 | 10.2% | Tensor pack/unpack; NCCL hidden during all of the above |
| **Compute-idle (stream 7 idle)** | **181** | **4.2%** | |
| ⑤ Waiting for NCCL to drain (exposed AG+RS) | 116.6 | 2.7% | RS tail (70ms): backward done, RS still running; AG head (35ms): step start |
| ⑥ Pure idle (no compute, no NCCL) | 65.0 | 1.5% | Kernel-to-kernel launch gaps (~5–50 µs each) + fwd→bwd boundary |
| **Total wall** | **4312** | **100%** | |
| *(NCCL concurrent — not in wall)* | *(3208.6)* | — | AG 1601ms + RS 1725ms − exposed 116ms |

### Improvement Potential

| Optimization | Mechanism | ms saved | New wall | Step speedup |
|---|---|---|---|---|
| **Flat parameter buffer** | Eliminate chunk_cat + split_with_sizes | ~424 ms | ~3888 ms | **~1.11×** |
| Torch.compile | Fuse elementwise/norm/RoPE, reduce launch gaps | ~50–100 ms | ~3788–3838 ms | ~1.12–1.14× |
| Fix AG startup gap (②) | Reduce Python overhead at step start | ~19 ms | ~3869 ms | ~1.01× |
| Async optimizer overlap | Move optimizer step off critical path | variable | — | — |

---

## Flat Parameter Buffer: Detailed Impact

FSDP2 currently packs/unpacks parameter shards at every forward/backward layer, adding two kernel types to the **serial critical path** on stream 7:

### chunk_cat_cuda_kernel (pre-ReduceScatter pack)

| Metric | Value |
|--------|-------|
| Kernels/step | ~81 (one before each RS) |
| Total GPU time/step | **237.8 ms** |
| Position | Immediately before each RS launch; RS fires 5–9 µs after chunk_cat completes |
| Critical path impact | **Full cost** — RS cannot start until chunk_cat finishes; fully serial |

### split_with_sizes_copy_out (post-AllGather unpack)

| Metric | Value |
|--------|-------|
| Kernels/step | ~160 (one after each AG) |
| Total GPU time/step | **202.4 ms** |
| Position | On compute stream 7; runs between AG completion and first GEMM |
| Critical path impact | **Full cost** — compute stream stalls on unpack before GEMM can start |
| % of compute stream | 4.9% of compute stream active time (4131 ms) |

### CatArrayBatchedCopy (flat-buffer copy, if present)

| Metric | Value |
|--------|-------|
| Kernels/step | ~3 (essentially absent) |
| Total GPU time/step | **0.005 ms** |
| Impact | Negligible in this profile |

### Net saving from flat buffer

```
chunk_cat eliminated:      237.8 ms/step (serial, directly on RS critical path)
split_with_sizes elim:     202.4 ms/step (on compute stream, blocks GEMM start)
CatArrayBatchedCopy cost:    ~0.0 ms/step  (baseline already ~0 without flat buffer)
                           ──────────────
Net saving:               ~440.2 ms/step   (10.2% of 4312 ms wall)

New wall:    4312 − 440 ≈ 3872 ms/step
Step speedup:  4312 / 3872 ≈ 1.114×

Additional RS startup-gap reduction:
  - Removing chunk_cat allows RS to launch sooner → up to 4.7 ms additional
  - (currently 4.7 ms startup gap / 3 steps = 1.6 ms/step)
```

**Summary**: Flat buffer eliminates 440 ms/step (~10.2% of wall) → **~1.11× step time speedup**, consistent with the MFSDP-vs-FSDP2 improvement (MFSDP sees 64 ms copy overhead vs FSDP2's 1,979 ms in the cp2 config, though that config has more parameters in flight).

---

## Raw Numbers Reference

All figures are for **device 0 (rank 0), steady-state steps 1–3** of iteration 7.

```
Step wall time:                 4312.2 ms
FSDP2 AG GPU time/step:         1600.6 ms  (37.1% of wall)
FSDP2 RS GPU time/step:         1724.6 ms  (40.0% of wall)
FSDP2 AG exposed/step:            35.2 ms  ( 0.8% of wall)   ← stall from AG
FSDP2 RS exposed/step:            81.4 ms  ( 1.9% of wall)   ← stall from RS
Total exposed NCCL/step:          116.6 ms  ( 2.7% of wall)
Compute stream active/step:     4131.2 ms  (95.8% of wall)
GEMM time/step:                 3186.5 ms  (73.9% of wall)
TE attention (fwd+bwd)/step:      226.5 ms  ( 5.3% of wall)
FSDP2 chunk_cat/step:             237.8 ms  ( 5.5% of wall)
FSDP2 split_with_sizes/step:      202.4 ms  ( 4.7% of wall)
FSDP2 copy overhead/step:         440.3 ms  (10.2% of wall)
```
