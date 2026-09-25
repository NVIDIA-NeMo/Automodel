# `nemo_automodel/_transformers/` — HuggingFace bridge

Adds to the repository root `AGENTS.md`.

This is the integration layer with HuggingFace Transformers.

| File | Purpose |
|---|---|
| `auto_model.py` | `NeMoAuto*` classes wrapping `PreTrainedModel` (distributed init, checkpoint hooks, backend dispatch) |
| `registry.py` | `MODEL_ARCH_MAPPING` and `_CUSTOM_CONFIG_REGISTRATIONS` |
| `capabilities.py`, `model_capabilities.py` | Per-model feature flags |
| `infrastructure.py` | Device-mesh construction and process-group lifecycle |
| `te_attention.py`, `kernel_patches.py`, `v4_patches/` | Transformer Engine paths and upstream patches |

## Registry

`registry.py` is the single place a model becomes reachable from
`NeMoAutoModelForCausalLM` and friends. The root file states the registration
requirement; the mechanical steps are in the
`nemo-automodel-model-onboarding` skill. A model that trains locally but is
missing from `MODEL_ARCH_MAPPING` will fail for every user going through the
`NeMoAuto*` entry points, and no existing test will catch it.

## Capabilities

Capability flags (`supports_fp8`, `supports_moe`, `has_combined_qkv`,
`supports_thd`, …) drive conditional logic across the framework, so a wrong
flag silently changes behavior far from where it is declared. When you add a
flag, give it a default that preserves current behavior for every model that
does not set it, and add the model to the capability tests under
`tests/unit_tests/_transformers/` and `tests/capability_registry/`.

Do not read `attn_implementation` directly to decide an attention path — the
backend config short-circuits it for native models.

## Upstream patches

`v4_patches/` and `kernel_patches.py` monkey-patch installed Transformers code.
Every patch must be version-guarded and must no-op cleanly on versions that do
not need it. Note the Transformers version you tested against in the PR.
