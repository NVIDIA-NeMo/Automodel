# Nemotron Secondary Patch Notes (Auxiliary Files)

This document covers the second patch set: files that are useful for performance,
stability, or diagnostics, but are separable from the core PP/EP + fixed-length
SQuAD correctness patch documented in `NEMOTRON_PP_EP_SQUAD_PATCH_NOTES.md`.

## Scope

Primary goals of this secondary patch set:
- improve startup/runtime robustness for distributed runs,
- improve performance paths used in benchmark experiments,
- improve MFU accounting fidelity for NemotronH,
- keep all changes either guarded or backward-compatible by default.

## File-Level Summary

### 1) `nemo_automodel/_transformers/auto_model.py`

What changed:
- Added a NemotronH-specific Liger patch path when `model_type == "nemotron_h"`.
- Added guarded monkey-patching for compatible components (RMSNorm and CE class).

Why:
- Upstream Liger does not expose a direct NemotronH adapter path in this stack.
- This keeps Liger optimization opt-in and model-safe for NemotronH.

Related env toggles:
- `NEMOTRONH_LIGER_RMSNORM` (default `1`)
- `NEMOTRONH_LIGER_CE` (default `1`)

### 2) `nemo_automodel/components/distributed/init_utils.py`

What changed:
- During NCCL process-group init, explicitly passes `device_id` to `init_process_group`.

Why:
- Avoids lazy device inference ambiguity and reduces init-time communicator warnings.

### 3) `nemo_automodel/components/distributed/parallelizer.py`

What changed:
- Added/expanded NemotronH parallelization behavior for EP-aware execution.
- Added EP mesh handling and local-expert ownership logic for NemotronH MoE paths.
- Added guarded runtime behavior for DeepEP and fallback execution modes.

Why:
- Enables EP to be truly active in NemotronH model execution, not just config-level.

### 4) `nemo_automodel/components/distributed/parallelizer_utils.py`

What changed:
- Refactored `fully_shard` invocation path into a helper that can safely pass:
  - `reshard_after_forward`
  - `ignored_params` (filtered to local module params)
- Threaded these options through recursive sharding helpers.

Why:
- Improves FSDP control for mixed EP/FSDP scenarios and avoids mis-scoped ignored params.

### 5) `nemo_automodel/components/distributed/pipelining/hf_utils.py`

What changed:
- Expanded HF pipeline forward patching to support NemotronH-like layer/block behavior.
- Added stage-level mask/position gating and NemotronH-specific mask handling paths.

Why:
- Fixes PP forward-path mismatches for NemotronH backbone/layer semantics.

### 6) `nemo_automodel/components/loss/masked_ce.py`

What changed:
- Added optional TP-aware parallel CE path for DTensor-sharded vocab logits.
- Supports native and TE-backed implementations (selected via env var).

Why:
- Avoids unnecessary full-gather behavior in TP settings and improves large-vocab CE efficiency.

Related env toggles:
- `NEMOTRONH_TP_PARALLEL_CE` (default off)
- `NEMOTRONH_TP_PARALLEL_CE_IMPL` (`native` / `te` / `auto` / `off`)

### 7) `nemo_automodel/components/utils/flops_utils.py`

What changed:
- Added NemotronH-specific FLOPs formula (`NemotronHConfig` mapping).
- Accounts for hybrid block types (Mamba, Attention, MLP, MoE) and LM head.

Why:
- Improves MFU/throughput analysis quality for NemotronH benchmarks.
- This affects reporting/analysis, not training semantics.

### 8) `nemo_automodel/recipes/llm/benchmark.py`

What changed:
- Uses scoped PP loss broadcast helper instead of manual point-to-point send/recv.
- Enables TF32 matmul/conv precision mode for benchmark runs on supported GPUs.

Why:
- Cleaner PP metric synchronization and better benchmark throughput defaults.

## Risk Profile

- Most changes are guarded or no-op unless specific features are enabled.
- Highest-risk areas are:
  - custom NemotronH EP execution logic (`parallelizer.py`),
  - TP-parallel CE path (`masked_ce.py`) when explicitly enabled.

## Recommended Validation for This Secondary Patch Set

1. Distributed startup smoke:
- multi-GPU init succeeds without NCCL device ambiguity warnings.

2. NemotronH PP/EP benchmark smoke:
- training step executes with finite loss and no collective desync.

3. TP CE A/B (if TP is enabled):
- compare baseline CE vs `NEMOTRONH_TP_PARALLEL_CE=1` for numerical sanity.

4. FLOPs/MFU accounting sanity:
- NemotronH benchmark reports stable MFU trends with expected relative ordering.

