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
`nsys stats --report nvtx_gpu_proj_sum <profile>.nsys-rep` breaks a step down by module. To profile a module in
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


## Viewing profiles

```bash
nsys-ui <profile_name>.nsys-rep
```
