# Nsight Systems Profiles (2026 September edition, DeepSeek-V4 / Moonlight-V4)

This folder documents the Nsight Systems profiles used in the Automodel guest lecture. The model is
[Moonlight-V4-16B-A3B](https://huggingface.co/akoumpa/Moonlight-V4-16B-A3B), the DeepSeek-V4 architecture at
Moonlight scale (16.5B total / 3.0B active parameters, 27 layers, 64 experts, hybrid CSA/HCA attention,
manifold-constrained hyper-connections). All runs start from random weights with mock data.

Profiles are not committed (they may contain environment data). Regenerate them with

```bash
./examples/scalable_ai/nsys_profiles/regenerate_profiles.sh            # portable profiles (any CUDA GPU)
RUN_HOPPER=1 ./examples/scalable_ai/nsys_profiles/regenerate_profiles.sh  # also the DeepEP / TileLang profiles
```

The script is the single source for what each profile runs -- `grep -n '^e2e ' regenerate_profiles.sh` shows the
resolved arguments. It exports `PYTHONPATH` and the tilelang/triton/inductor cache directories first; without
those, tilelang fails to import and Automodel silently falls back to the transformers model class, so a profile
that looks fine is measuring the wrong implementation. Source those exports before running any command by hand.

Only end-to-end profiles are produced here. Per-module timings can be read straight out of these reports --
`--nvtx true` annotates every submodule's forward and backward, so
`nsys stats --report nvtx_kern_sum <profile>.nsys-rep` breaks a step down by module. `./extract_stats.sh`
runs that report and five others over the whole ladder and writes one CSV per report per profile into `stats/`,
which is how the tables below were produced; it skips a profile only when every report's CSV is already present,
so adding a report re-exports rather than serving a stale set. To profile a module in
isolation instead (single process, no distributed traffic, one backend at a time), run
`examples/scalable_ai/profile_layer.py` directly; see the parent `examples/scalable_ai/README.md`.

## Why the small configs look the way they do

- `--model.config.num_hidden_layers N` keeps the first N entries of the 27-entry schedule
  `[0, 0, 4, 128, 4, 128, ...]`: 2 layers = sliding-window only, 3 layers = adds one CSA layer, 4 = adds an HCA layer.
- The transformers implementation of DeepSeek-V4 is inference-oriented: in training mode its CSA layers gather the
  per-query top-k entries into an `S x k` key axis and its compressed entries carry no causal mask. The HF baseline
  is therefore profiled on the 2 sliding-window layers only; the NeMo Automodel run uses the same 2 layers for a
  like-for-like comparison, plus a 3-layer run that includes CSA.
- `attn: tilelang` (TileLang sparse attention + indexer + Sinkhorn kernels) and DeepEP need Hopper-class GPUs; the
  vendored kernels request 150-230 KB of shared memory per block.

## End-to-end benchmark profiles (`nemo_automodel/recipes/llm/benchmark.py`)

`--nvtx true` turns on `nemo_automodel.autonvtx`, which recursively wraps every submodule's forward and backward
in an NVTX range named `<child>: <ClassName>`, so the timeline reads as `DeepseekV4Attention`, `DeepseekV4Indexer`,
`MoE`, `GroupedExperts` rather than as anonymous kernels. It is off by default (one hook pair per module is not
free), and the recipe already emits `iteration_<i>_ga_step_<j>` ranges around each micro-batch either way. The
hooks are installed once at setup, so they are live for every step rather than only the captured window.

Every profile is the same invocation with a different config and stage flags:

```bash
nsys profile --force-overwrite true --trace=cuda,nvtx --cuda-memory-usage=true --output=<name>.nsys-rep \
  torchrun --nproc-per-node 2 nemo_automodel/recipes/llm/benchmark.py --config <config.yaml> \
  --benchmark.nsys_start 3 --benchmark.nsys_end 5 --step_scheduler.max_steps 6 --benchmark.warmup_steps 1 \
  --step_scheduler.global_batch_size 4 --step_scheduler.local_batch_size 1 --dataset.seq_len 1024 --nvtx true \
  <stage flags>
```

Besides the stage ladder below, the script produces `moonlight_v4_torch_small` -- the NeMo Automodel native model
on 2 sliding-window layers at `ep_size 1`, the like-for-like partner to stage 1's stock transformers run.

## Stage ladder (`stage1_*` .. `stage5_*`)

One optimization per step, everything else held fixed, so each profile differs from the previous one by exactly
one backend knob and every delta is attributable to it. All stages run the same 3-layer model (the schedule's
first three entries: SWA, SWA, CSA) at `ep_size 2`, except stage 1 -- stock transformers cannot train CSA layers,
so it runs the 2 sliding-window layers only.

| stage | what changes from the previous stage | config | needs |
| --- | --- | --- | --- |
| `stage1_hf_ootb` | baseline: stock transformers | `moonlight_v4_16b_hf.yaml` | |
| `stage2_am_expertloop` | NeMo Automodel native model (`experts: torch`, the per-expert loop) | `..._torch.yaml` | |
| `stage3_groupedgemm` | `experts: torch_mm` -- one grouped GEMM instead of a GEMM per local expert (32 of the 64 at `ep_size 2`) | `..._torch.yaml` | |
| `stage4_deepep` | `dispatcher: deepep` instead of the torch all-gather dispatcher | `..._torch.yaml` | `deep_ep` |
| `stage5_tilelang` | `attn: tilelang` -- sparse attention + indexer kernels instead of eager | `..._tilelang_deepep.yaml` | Hopper, `tilelang`, `tile_kernels` |

Stages 1-3 run anywhere; 4 and 5 are produced only with `RUN_HOPPER=1`. Each stage passes only the knob that
differs from the stage above it and leaves every other backend to its config, so stages 3 and 5 pass no backend
flags at all -- they are their config's defaults.

**Stage 5's backends are exactly those of `configs/pretrain_moonlight_v4_16b.yaml`**, so the ladder converges on
the recipe V4 is actually trained with rather than on an "everything on" configuration. `rms_norm` is deliberately
not a rung: the training recipe keeps `torch_fp32` because V4 holds the attention sinks, compressor position
biases, mHC mixers and `lm_head` in fp32. Use `profile_layer.py --layer rmsnorm` to profile that kernel alone.

Two settings are benchmark conventions rather than training settings, shared with the production benchmark configs
under `examples/llm_benchmark/`. `num_hash_layers: 0` is set by all three configs. `fake_balanced_gate: true`
removes expert load-imbalance noise and applies to the Automodel stages only -- it is a `BackendConfig` field, and
`moonlight_v4_16b_hf.yaml` has no `backend:` block, so stage 1 routes with its own random-init gate. The
pretraining recipe sets `fake_balanced_gate: false` and leaves the hash layer on.


### What the ladder measures

Per-module numbers from `nsys stats --report nvtx_kern_sum <stage>.nsys-rep`, which sums **kernel execution time**
inside each NVTX range. Use this report, not `nvtx_gpu_proj_sum`: the latter's `Proj *` columns are the *span* of a
range on the GPU timeline, gaps included, so a module whose kernels are unchanged but whose launches spread out
reads as slower. Scored with spans, this ladder overstated every win it found and hid two regressions entirely.

**Not every module range covers its backward pass.** `autonvtx` opens a backward range for each module, but when a
module's backward runs inside a custom autograd Function the kernels land outside that range. Measured on
`stage2`, one `iteration_5_ga_step_0`: the two forward-thread `self_attn` ranges hold 16.2 / 16.6 ms of wall time
and 10.3 / 12.3 ms of kernel time over ~700 kernels each, while the two backward-thread ranges of the same name
hold 0.02 ms and under 10 kernels. `self_attn` and `indexer` are therefore **forward-only**; `mlp: MoE`,
`experts`, `attn_hc` and `compressor` cover both passes.

| module (kernel time, ms, whole capture, both ranks) | stage2 loop | stage3 grouped | stage4 DeepEP | stage5 TileLang |
| --- | ---: | ---: | ---: | ---: |
| `experts` | 1964.9 | 203.4 | 5535.6 | 5512.2 |
| `mlp: MoE` | 1979.9 | 218.6 | 5551.3 | 5527.9 |
| `self_attn` (fwd only) | 131.7 | 196.9 | 142.1 | 158.4 |
| `indexer` (fwd only) | 18.7 | 16.9 | 18.1 | 24.3 |
| `compressor` | 34.9 | 31.8 | 36.7 | 53.3 |
| `attn_hc` (mHC) | 69.2 | 69.3 | 69.4 | 22.3 |

#### The noise floor, and which deltas the ladder can resolve

No rung before stage 5 touches attention, so stages 2-4 are three replicates of the same attention work and their
spread *is* this ladder's noise floor -- per module, because it differs per module by two orders of magnitude:

| module | stage2-4 spread | resolves to | stage5 | verdict |
| --- | --- | ---: | ---: | --- |
| `attn_hc` | 69.2 / 69.3 / 69.4 | 0.1% | 22.3 | **3.11x**, far outside the spread |
| `indexer` | 18.7 / 16.9 / 18.1 | ~5% | 24.3 | **0.74x**, a real regression |
| `compressor` | 34.9 / 31.8 / 36.7 | ~7% | 53.3 | **0.69x**, a real regression |
| `self_attn` | 131.7 / 196.9 / 142.1 | ~20% | 158.4 | **not resolvable** -- inside the spread |

Quote a delta only when it clears its module's spread. `self_attn` does not: 158.4 sits between the 131.7 and
196.9 this ladder produces for stages that changed nothing about attention.

| rung | component it targets | speed-up |
| --- | --- | ---: |
| stage2 -> stage3 (grouped GEMM) | `experts` | **9.66x** |
| stage2 -> stage3 (grouped GEMM) | `mlp: MoE` | 9.06x |
| stage4 -> stage5 (TileLang) | `attn_hc` Sinkhorn | **3.11x** |
| stage4 -> stage5 (TileLang) | `indexer`, forward only | **0.74x** |
| stage4 -> stage5 (TileLang) | `compressor` | **0.69x** |
| stage4 -> stage5 (TileLang) | `self_attn`, forward only | not resolvable |

**The DeepEP rung cannot be scored from this table.** `nvtx_kern_sum` has no time filter, and DeepEP JIT-compiles
during the first captured iteration, so stages 4-5 are almost entirely warmup:

```
kernel time, iteration_0_ga_step_0 -> iteration_5_ga_step_0
  stage2    140.3 ->  56.6 ms
  stage3    239.7 ->  39.5 ms
  stage4   2777.0 ->  46.7 ms     <- 59x the steady step
  stage5   2849.5 ->  31.1 ms
```

The attention-side rows survive that because their kernel counts are identical across stages 2-4 (8388, 1956,
32004 and 4780 respectively), so warmup contributes the same fraction everywhere and cancels in the ratio. The MoE
rows do not. Scoring DeepEP needs a capture that excludes iteration 0 -- `--benchmark.nsys_start 4` in
`regenerate_profiles.sh` -- which means re-running those two profiles.

**`attn: tilelang` is one config key but three kernel families.** The kernels that appear only in `stage5` are
`sparse_mqa_bwd_kernel` (sparse attention), `tl_indexer_fwd_kernel` (indexer) and the `tile_kernels` Sinkhorn
kernels (hyper-connections). At this shape that bundle is **one large win and two regressions**: Sinkhorn 3.11x,
indexer 0.74x, compressor 0.69x. `attn_hc`'s kernel count drops 32004 -> 3780 for the win; `indexer` and
`compressor` keep identical kernel counts and simply cost more.

The sparse-attention kernel the rung is named for cannot be scored here at all -- `self_attn` is below the noise
floor, and `sparse_mqa_bwd_kernel` runs outside the `self_attn` range anyway, so TileLang's backward win is not in
these numbers even in principle. The only trustworthy attention figure is the isolated one in the parent README:
at sequence 4096, forward 8.285 -> 9.077 ms (0.91x) and backward 23.593 -> 8.693 ms (2.71x), for **1.79x** overall.
Score the attention rung from `profile_layer.py`, not from here.

## Bottlenecks and where to look next

What the current evidence says is limiting, in priority order.

1. **The per-expert loop is HBM-bandwidth bound, not compute bound.** In one steady-state `stage2` gradient-
   accumulation step (`iteration_5_ga_step_0`, 185 ms of wall time on one rank), `CUDAFunctor_add` and
   `FillFunctor` account for 137 ms of GPU time. Their launch geometry (`grid 90112 x block 128 x 8 elements`) is
   exactly `64 experts x 1024 tokens x 1408 intermediate`: the loop zero-fills and scatter-adds a full `[E, T, I]`
   buffer per expert. At ~1.11 GB moved per 361 us launch those kernels already run near HBM3 peak, so there is no
   kernel to tune -- only the traffic to remove, which is what `experts: torch_mm` does.
2. **`_permute_kernel` / `_unpermute_kernel` are the largest remaining MoE overhead** in the DeepEP stages. Fusing
   the token gather/scatter into the grouped-GEMM epilogue would keep it off HBM entirely; CUTLASS exposes this
   through the collective epilogue builder's fusion operation, and its group GEMM (variable-size, one kernel) is
   the documented pattern for the expert GEMM itself.
3. **FSDP2's AllGather is essentially never overlapped: 93-97% of it is exposed.** Measured with
   [`nsys-overlap`](https://gitlab-master.nvidia.com/zhiyul/nsys-overlap)'s 3-bucket decomposition, which assigns
   every nanosecond of merged collective wall time to hidden-by-deep-compute, hidden-by-light-compute, or truly
   exposed (compute stream idle). Steady-state windows only (`iteration_3..5`, both gradient-accumulation steps),
   both ranks:

   | stage | AG exposed, rank 0 | AG exposed, rank 1 | RS exposed, rank 0 | RS exposed, rank 1 |
   | --- | ---: | ---: | ---: | ---: |
   | stage3 grouped GEMM | 96.2% | 97.0% | 56.0% | 59.7% |
   | stage4 DeepEP | 92.9% | 96.4% | 64.8% | 54.6% |
   | stage5 TileLang | 96.8% | 95.9% | 52.8% | 53.4% |

   ReduceScatter is roughly half hidden; AllGather is not hidden at all. The model gives the prefetcher almost
   nothing to hide it behind -- in `stage3` the compute stream holds 503 ms of kernel wall time, but GEMM-shaped
   work is a small fraction of it, the rest being the elementwise traffic item 1 describes. The same pathology is
   documented for Qwen3-MoE-30B in the `nsys-overlap` README, where registering expert GEMM streams as valid
   overlap partners for the AG prefetcher was the identified fix. Worth testing here, though note this model runs
   its experts on the main compute stream rather than on dedicated ones, so the mechanism differs.

4. **`sparse_mqa_bwd_kernel` is the single largest attention kernel** once TileLang is on. Whether it is
   bandwidth-, occupancy- or latency-limited is a kernel-internal question Nsight Systems cannot answer -- capture
   it with Nsight Compute (Speed of Light plus memory workload analysis) before changing it.

### What the exposed-communication numbers do and do not support

The exposed percentages above were checked four ways before being written down, because a single overlap number is
easy to get wrong:

- **Not a warmup artifact.** Restricting to `iteration_3..5` changes nothing; `nsys-overlap` run over the whole
  capture gives 92.5 / 94.3 / 93.7 / 97.6% for AG against 95.2 / 96.2 / 92.9 / 96.8% windowed.
- **Not an artifact of one rank.** Both ranks agree, on different compute streams (7 and 21).
- **Not an artifact of one implementation.** An independent interval-intersection written directly against
  `CUPTI_ACTIVITY_KIND_KERNEL` reproduces `nsys-overlap` within ~2 points on every stage.
- **Not a stream-classification error.** The tool's own README warns its heuristic can misclassify MoE expert
  streams. Here each profile has one dominant compute stream (410-503 ms) with the rest being NCCL, DeepEP
  dispatch, and optimizer streams, so there is no hidden expert-GEMM stream to miss.

Two things the numbers do **not** support:

- **"Zero AllGather is hidden behind GEMM."** `nsys-overlap` reports bucket A at 0.0%, but it classifies GEMM by
  the kernel-name pattern `%nvjet_sm%`, which does not match this model's `sm90_xmma_gemm_*`, CUTLASS,
  `torch._grouped_mm` or TileLang kernels. The A-versus-B split is therefore unreliable here. Only bucket C --
  comm time with the compute stream fully idle -- is robust, since it depends on kernel presence rather than
  kernel naming.
- **Absolute exposed milliseconds.** Exposed AG ranges over 12.9-42.9 ms between stages, including a 3x swing
  between two stages that share a dispatcher, so only the *fraction* is stable enough to quote.

Scope: 2 ranks, one node, `ep_size 2`, sequence 1024. Whether AllGather stays this exposed at EP 8 across nodes is
untested here.

Gaps in this profiling setup, which bound how far the above can be trusted:

- **Memory is not captured.** `regenerate_profiles.sh` passes `--cuda-memory-usage=true`, but
  `CUDA_GPU_MEMORY_USAGE_EVENTS` is empty in the stage reports. Peak memory is the headline of the parent README's
  reference tables (22.0 -> 14.2 GB, and only the TileLang path fits micro-batch 8) and none of it can be read
  here; those numbers come from `max_memory_allocated` in the bench runs instead.
- **The ladder is profiled at a launch-bound operating point.** At the `E2E_COMMON` shape above, wall-clock gains
  (1.92x for grouped GEMM) exceed GPU-time gains (1.50x), so the ladder flatters operation-count reductions and
  understates bandwidth ones. Re-running it at the journey shape (4 layers, sequence 2048, micro-batch 4, 8 GPUs)
  would reorder the rungs.
- **DeepEP is measured where it cannot win.** `ep_size 2` on a single node, plus the `fake_balanced_gate`
  convention noted above, removes exactly the load imbalance and cross-node traffic DeepEP exists to fix -- hence
  ~1.00x end to end here, against 1.14x at EP 8 in the parent README's journey table. Its per-module effect is not
  measurable from these captures either, for the warmup reason given above.
- **Two rungs are not single-knob deltas.** Stage 1 runs 2 layers for the reason given above, so `stage1 ->
  stage2` is not a valid delta; `moonlight_v4_torch_small` is the like-for-like control and is not yet in the
  ladder's own stats. And `attn: tilelang` bundles three kernel families; splitting it into three sub-rungs would
  make that delta attributable.

## Viewing profiles

```bash
nsys-ui <profile_name>.nsys-rep
```
