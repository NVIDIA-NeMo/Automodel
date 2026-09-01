# `nemo_automodel/components/models/` — model implementations

Adds to the repository root `AGENTS.md`.

**Load the `nemo-automodel-model-onboarding` skill before adding or changing a
model family.**

## Directory layout

Each model lives under `nemo_automodel/components/models/<name>/` and contains:

| File | Purpose |
|---|---|
| `model.py` | Model class (inherits `PreTrainedModel` + `HFCheckpointingMixin`) |
| `state_dict_adapter.py` | Weight key mapping between HF and NeMo formats |
| `config.py` (optional) | Custom config class if the HF config is insufficient |
| `layers.py` (optional) | Custom layer implementations |
| `rope_utils.py` (optional) | Model-specific RoPE variants |

## Inheritance

- All models inherit from `PreTrainedModel` and `HFCheckpointingMixin`.
- MoE models additionally inherit `MoEFSDPSyncMixin` (see
  `nemo_automodel/components/moe/fsdp_mixin.py`) for correct expert gradient
  synchronization
  under FSDP2. Omitting it produces wrong gradients, not a crash.

## Registration

Register the model as described in the root `AGENTS.md`; the mechanics live in
`nemo_automodel/_transformers/registry.py`.

## Combined projections

Combined projections (fused QKV, fused GateUp) use **interleaved layout** so
that tensor-parallel sharding splits evenly across heads/experts. Do not change
the interleave order without working through the TP implications — a wrong
order still loads and still trains, it just trains a different model.

## Backends

`BackendConfig` selects the attention, linear, normalization, RoPE, and expert
dispatch implementations. Backend selection comes from the YAML config and is
threaded through model construction; **individual layers must never hard-code a
backend choice** or branch on `attn_implementation` directly.

## Parity

A new or modified model needs numerical parity evidence against the HF
reference. Follow the `parity-testing` skill — do not hand-tune a tolerance
until a comparison passes.
