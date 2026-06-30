# Model packages

This directory owns model-family behavior. A package under `models/<family>/`
is the only place that should know its module layout, checkpoint naming,
parallelization exceptions, or remote-code quirks.

## Standard package shape

Not every package needs every file. Keep a file only when the model family
owns that behavior.

| File | Ownership |
| --- | --- |
| `__init__.py` | Lightweight package setup and any import-time integration hook. Do not import heavy model code unnecessarily. |
| `model.py` | Native `nn.Module` / `PreTrainedModel` implementation. Export `ModelClass = ...` at the bottom for registry discovery. |
| `config.py` or `configuration.py` | A config type or config-specific compatibility code when the upstream config is insufficient. |
| `layers.py` | Family-specific attention, MLP, normalization, MoE, or block layers. |
| `rope_utils.py` | Family-specific RoPE scaling or position-encoding behavior. |
| `state_dict_adapter.py` | Checkpoint key/layout translation between upstream and NeMo model structures. |
| `parallelizer.py` | Model-specific TP plans or FSDP strategy behavior. This is the home for concrete module paths and model-specific distributed policy. |
| `cp.py`, `cp_attention.py`, `cp_batch.py` | Context-parallel behavior that is unique to that family. |
| `mtp.py` | Multi-token-prediction heads or stage customization. |
| `processor.py`, `processing.py`, `vision_encoder.py`, `autoencoder.py` | Multimodal input or vision/audio component ownership. |
| `optimized_kernels.py`, `kernels/` | Optional family-specific kernels, guarded so their absence cannot break package import. |

`model.py` is not required for an upstream-only model. Such a directory may
contain only a `parallelizer.py`; it is still the owner of that family’s local
parallelization policy.

## Parallelization boundary

`components/distributed/` resolves and applies a plan. It must not import
individual model packages or encode model paths.

- Put concrete paths such as `model.layers.*` or
  `model.language_model.layers.*` in that family’s `parallelizer.py`.
- Put reusable TP styles and decoder-plan mechanics in
  `models/common/tp_plan.py`. Its `DecoderTPPaths` input is declared by the
  model sidecar; it must not carry a default model prefix.
- A sidecar exposes `get_tp_plan(model, *, sequence_parallel=False)` and, when
  needed, `get_parallelization_strategy()`.
- Do not add a global TP registry. The adapter finds the owner through the
  model registry and conventionally probes `<model-package>.parallelizer`; for
  upstream-only models it falls back to `config.model_type`.
- Keep sidecar imports lazy. A native model may install a small factory that
  imports its sibling `parallelizer.py` only when parallelization is requested.

## Directory map

| Directory | Primary role |
| --- | --- |
| `common/` | Cross-family primitives: backend/config utilities, checkpoint mixins, packing, bidirectional helpers, and shared TP mechanics. Never put family paths or architecture dispatch here. |
| `bagel/` | Unified multimodal model with packed Qwen2 and SigLIP/autoencoder support. |
| `baichuan/` | Native Baichuan decoder and its MLP-only TP sidecar. |
| `deepseek_v3/`, `deepseek_v32/`, `deepseek_v4/` | Native DeepSeek generations; V4 additionally owns CP, FSDP, MTP, kernels, and its distributed sidecar. |
| `diffusion_gemma/` | Native block-diffusion Gemma model, layers, masks, checkpoint adapter, and FSDP/TP behavior. |
| `ernie4_5/`, `ling_v2/`, `hy_v3/`, `hy_mt2/`, `mimo_v2_flash/`, `minimax_m2/`, `step3p5/`, `step3p7/` | Native dense decoder families with their model-specific layers/config/checkpoint code where present. |
| `gpt2.py` | Legacy single-file GPT-2 implementation; do not create a package merely for symmetry. |
| `gpt_oss/`, `glm4_moe/`, `glm4_moe_lite/`, `glm_moe_dsa/`, `qwen3_moe/`, `qwen3_next/`, `qwen3_5_moe/`, `qwen3_omni_moe/`, `qwen3_vl_moe/` | MoE or hybrid-MoE families. Keep expert, router, and context-parallel policy local. |
| `llama/`, `qwen2/`, `mistral3/` | Native dense decoder baselines. Their sibling sidecars own their decoder TP paths. |
| `llama_bidirectional/`, `ministral_bidirectional/` | Retrieval/bidirectional variants of the decoder families. |
| `llama_nemotron_vl/`, `llava_onevision/`, `kimivl/`, `kimi_k25_vl/`, `minimax_m3_vl/`, `mistral3_vlm/`, `mistral4/`, `nemotron_omni/`, `nemotron_parse/`, `qwen2_5_omni/` | Vision, audio, parsing, or omni wrappers. Keep processors, vision encoders, and multimodal checkpoint adaptation here. |
| `gemma4_moe/`, `gemma4_drafter/` | Gemma4 MoE and drafter implementations; each keeps its own model and sidecar behavior. |
| `nemotron_v3/` | Native Nemotron-H V3 implementation including cache, MTP, layers, and custom strategy sidecar. |
| `falcon_h1/`, `gemma3/`, `mixtral/`, `nemotron_labs_diffusion/`, `nemotron_nas/`, `phi/`, `phi3/`, `qwen3/` | Upstream or remote-code families with model-local TP sidecars but no native `model.py`. Add native implementation files here only when Automodel takes ownership of the architecture. |
| `qwen3_5/` | Native Qwen3.5 dense model and sidecar; related MoE/Omni variants remain in their dedicated packages. |

## Adding or changing a model package

1. Start with the nearest family in this directory, not a generic distributed
   conditional.
2. Add the architecture to `_transformers/registry.py` only when Automodel owns
   a native implementation. Upstream-only sidecars rely on the model-type
   convention instead.
3. Put checkpoint conversion in `state_dict_adapter.py`, never in training or
   distributed infrastructure.
4. Put model-only TP paths and special FSDP behavior in `parallelizer.py`; add
   focused CPU tests beside the distributed unit tests.
5. Reuse `common/` only for genuinely layout-independent code. If it needs a
   model path, config-model-type branch, or architecture name, it belongs in a
   family directory.
