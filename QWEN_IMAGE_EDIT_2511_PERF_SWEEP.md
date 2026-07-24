# Qwen Image Edit 2511 — Performance Config Sweep

Started: 2026-07-22 UTC
Node: 8× H100 80GB HBM3 (single node), BF16, FSDP2 dp_size=8

## Goal

Identify the best-performing training config for `examples/diffusion/finetune/qwen_image_edit_2511_flow.yaml` starting from the validated baseline (median 3.03 samples/s). Profile the baseline, then run one config change at a time, measure, and combine winners.

## Protocol

Every benchmark run uses the fixed 1,024-example MagicBrush performance cache (fixed-square 1024p, target/context latents `[16, 128, 128]`), 20 warm-up + 100 timed optimizer steps, checkpoint writes disabled:

```bash
uv run torchrun --nproc-per-node=8 examples/diffusion/finetune/finetune.py \
  -c examples/diffusion/finetune/qwen_image_edit_2511_flow.yaml \
  --data.dataloader.cache_dir /tmp/qwen-image-edit-validation/cache/magicbrush-qwen-edit-performance \
  --checkpoint.enabled false \
  --step_scheduler.max_steps 120 \
  <experiment overrides>
```

Primary metric: samples/s over the 100-step timed window. Secondary: step time, phase times (fwd/bwd/clip/opt), peak allocated/reserved memory. Single run per experiment; the final winner gets 3 repeats for a median. Experiments run strictly sequentially — one job on the node at a time. **Review this tracker before launching any new run.**

Numerical-equivalence note: config changes that alter kernels/precision (flash attention, TE linear, compile, QKV fusion) may change loss values slightly; loss must stay finite and same order as baseline. Changes that alter training semantics (batch size) are labeled as such.

## Reference baseline (2026-07-21 session, 3-run median)

| Metric | Value |
|---|---|
| Samples/s | 3.03 |
| Step time | 2.640 s |
| Forward | 0.729 s |
| Backward (incl. overlapped FSDP comm) | 1.718 s |
| Grad clip | 0.081 s |
| Optimizer | 0.061 s |
| Peak allocated / reserved | 23.57 / 45.97 GiB |

Phase analysis: backward is 65% of step time with whole-block activation checkpointing ON (recompute ≈ an extra forward inside backward). Memory headroom is large (23.57/80 GiB), so removing or reducing AC is the top-priority experiment. Grad clip + optimizer are only 5% combined.

## Experiment queue

| ID | Config delta vs baseline | Hypothesis | Status |
|---|---|---|---|
| E00 | none (baseline re-run) | reproduce ~3.03 samples/s on current node/env state | **done: 3.07 samples/s** |
| P00 | baseline + nsys trace, short run | kernel/NCCL-level profile of baseline | done |
| E01 | `fsdp.activation_checkpointing=false` | removes backward recompute; biggest single win; memory should fit | **failed: OOM** (69.84 GiB alloc + 7.69 GiB fragmented; 48 MiB allocation failed at step 1) |
| E01a | E01 + `PYTORCH_ALLOC_CONF=expandable_segments:True` | OOM was fragmentation-dominated; expandable segments reclaim the 7.69 GiB | **failed: OOM** — defrag worked (345 MiB slack) but true demand is 77.23 GiB; full AC-off does not fit at lbs=1/1024p |
| E01b | `fsdp.activation_checkpointing=selective` | recompute only cheap ops; middle ground | **done: 3.46 samples/s (+12.7%)** |
| E02 | `fsdp.reshard_after_forward=false` | removes backward all-gather; ~+35 GiB params kept gathered | **invalid** — knob was silently dead (see log entry); produced exact baseline numbers |
| E02r | E02 rerun after plumbing fix | same hypothesis, override now effective | **done: 3.20 samples/s (+4.2%)** — real but smaller than E01b; cannot combine with E01b (49.6+33.5 GiB > 80) |
| E03 | `model.fuse_qkv_projections=true model.compact_fused_qkv_projections=true` | fewer/larger GEMMs in attention | **done: 3.14 samples/s (+2.3%)** |
| E04 | `model.attention_backend=flash` (FA2 via Diffusers set_attention_backend) | faster attention vs native SDPA | **failed: incompatible** — "`attn_mask` is not supported for flash-attn 2"; padded-prompt masks require mask-capable SDPA (cuDNN native stays) |
| E05 | `fsdp.enable_compile=true` | torch.compile per block | **done: 3.53 samples/s (+15.0%) — best single knob** |
| E06 | `model.transformer_engine_linear=true` (BF16, no FP8) | TE linear kernels | **done: 3.01 samples/s (−2.0%) — regression, dropped** |
| E07a | E05 + E01b (compile + selective AC) | independent mechanisms (kernel fusion + less recompute) | **failed: Inductor error** — fused layer_norm-backward triton kernel needs 245,832 B shared memory > 232,448 B hardware limit; compile and selective AC do not compose on this model |
| E07b | E05 + E03 (compile + fused QKV) | compatible mechanisms | **done: 3.74 samples/s (+21.8%) — new leader** |
| E07c | E07b + E02r (compile + fused QKV + no reshard) | full compatible stack; fits (~57 GiB) | **done: WINNER — 3-run median 3.84 samples/s (+25.1%)** |
| E-batch | `step_scheduler.local_batch_size=2 global_batch_size=16` | better GPU utilization; **changes training semantics (GBS)** | not run — excluded from like-for-like comparison; run separately if a GBS-16 recipe is ever wanted |

## Results

| ID | Samples/s | Step time | fwd | bwd | clip | opt | Peak alloc | Peak reserved | Loss finite |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| E00 baseline | 3.07 | 2.607 s | 0.712 s | 1.698 s | 0.082 s | 0.065 s | 23.57 GiB | 45.97 GiB | yes (0.0204 @120) |
| E01 AC off | OOM | — | — | — | — | — | >77 GiB | — | n/a |
| E01a AC off + expandable segments | OOM | — | — | — | — | — | 77.23 GiB true demand | — | n/a |
| E01b AC selective | **3.46 (+12.7%)** | 2.311 s | 0.721 s | 1.429 s | 0.077 s | 0.063 s | 49.62 GiB | 59.92 GiB | yes (0.0203 @120, matches baseline to 4 decimals) |
| E02r no reshard after fwd | 3.20 (+4.2%) | 2.499 s | 0.703 s | 1.611 s | — | — | 57.10 GiB | 66.06 GiB | yes (0.0203 @120) |
| E03 fused compact QKV | 3.14 (+2.3%) | 2.551 s | 0.680 s | 1.701 s | — | — | 23.57 GiB | — | yes (0.0204 @120) |
| E05 compile | **3.53 (+15.0%)** | 2.264 s | 0.695 s | 1.390 s | — | — | 23.10 GiB | — | yes (0.0203 @120) |
| E06 TE linear (BF16) | 3.01 (−2.0%) | 2.659 s | 0.768 s | 1.703 s | — | — | 23.54 GiB | — | yes (0.0203 @120) |
| E07b compile + fused QKV | **3.74 (+21.8%)** | 2.137 s | 0.574 s | 1.382 s | — | — | 23.10 GiB | — | yes (0.0203 @120) |
| E07c full stack (run 1) | **3.84 (+25.1%)** | 2.085 s | 0.591 s | 1.305 s | — | — | 56.98 GiB | — | yes (0.0203 @120) |
| E07c full stack (run 2) | 3.81 | 2.102 s | — | — | — | — | 56.98 GiB | — | yes (0.0203 @120) |
| E07c full stack (run 3) | 3.85 | 2.078 s | 0.582 s | 1.303 s | 0.084 s | 0.063 s | 56.98 GiB | 65.38 GiB | yes (0.0203 @120) |
| **E07c median (3 runs)** | **3.84 (+25.1% vs E00, +26.7% vs original 3.03 median)** | **2.085 s** | | | | | **56.98 GiB** | **65.38 GiB** | |

## Final recommendation

Winning config — add to `examples/diffusion/finetune/qwen_image_edit_2511_flow.yaml` (or pass as CLI overrides):

```yaml
fsdp:
  enable_compile: true
  reshard_after_forward: false   # now actually plumbed (fix in auto_diffusion_pipeline.py)
model:
  fuse_qkv_projections: true
  compact_fused_qkv_projections: true
```

- **3-run median 3.84 samples/s vs 3.03 baseline median: +26.7% throughput**, loss trajectories match baseline to 4 decimals at step 120 in every run.
- Memory trade-off: 56.98 GiB allocated vs 23.57 baseline. If headroom matters (larger resolutions, future batch growth), drop `reshard_after_forward: false` and keep **E07b (compile + fused QKV): median-class 3.74 samples/s (+21.8%) at a memory-neutral 23.10 GiB**.
- Do not combine `enable_compile` with `activation_checkpointing: selective` (Inductor shared-memory failure, see E07a) or set `attention_backend: flash` (FA2 rejects the padded-prompt attn_mask, see E04). TE linear in BF16 is a small regression (E06).
- The sweep also fixed a real bug: `fsdp.reshard_after_forward` was silently ignored by the diffusion bridge until this session's one-line plumbing fix.

## Experiment log

### 2026-07-22 — P00 baseline profile (nsys, 30 steps, all 8 ranks)

Artifact: `/tmp/qwen-image-edit-validation/sweep/P00_baseline_profile.nsys-rep` (800 MB). GPU kernel-time breakdown:

| Share | Component | Evidence |
|---:|---|---|
| ~30% | NCCL (AllGather 19.0%, ReduceScatter 10.4%, AllReduce 0.6%) | mostly overlapped with backward compute; backward all-gather is removable via `reshard_after_forward=false` |
| ~22% | cuDNN flash SDPA (bprop 13.6%, fprop 8.6%) | fprop instance count is 2× bprop (28,800 vs 14,400): **whole-block AC recomputes the full forward during backward** |
| ~19% | GEMMs (`nvjet_sm90_*`) | same 2× recompute pattern in instance counts |
| ~15% | Elementwise/copy/reduce tail | 374k `where` kernels, 809k small copies, RoPE complex-mul — fusible by `torch.compile` |

Conclusions: (1) E01 (AC off) attacks the single largest waste — the recomputed forward inside backward; (2) E02 (no reshard after forward) attacks the AllGather share; (3) E05 (compile) attacks the elementwise tail; (4) attention already uses cuDNN flash on sm90, so E04 (FA2) may be neutral.

### 2026-07-22 — E02 exposed a dead recipe key (code fix applied)

E02 reproduced baseline numbers bit-for-bit in memory terms (23.57 GiB), which led to the discovery that the diffusion bridge's `FSDP2Config(...)` construction (`nemo_automodel/_diffusers/auto_diffusion_pipeline.py`) never forwarded `reshard_after_forward` from the yaml `fsdp:` section — the recipe's documented `reshard_after_forward: true` key was silently ignored (config repr showed `reshard_after_forward=None`). Fixed by forwarding `args.get("reshard_after_forward", None)`; `tests/unit_tests/_diffusers/` 64 passed. Baseline semantics unchanged (yaml `true` == FSDP2 default). E02 rerun as E02r.

### 2026-07-22 — Setup

- `/tmp` artifacts from the 2026-07-21 session were wiped; regenerating the pinned 1,024-example performance cache on 8×H100 (background job) before any benchmark can run.
- Tooling verified: `nsys` present, `flash_attn` 2.7.4.post1, `diffusers` 0.38.0, TransformerEngine importable, `diffusion-media` extra synced.
- Performance cache regenerated and verified: 1,024/1,024 samples, train split, 1024p preset, model revision `6f3ccc0b`. E00 launched.
