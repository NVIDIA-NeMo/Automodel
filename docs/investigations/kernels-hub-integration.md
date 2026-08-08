# Kernels Hub Integration Investigation

**Branch:** `kashif/investigate/kernels-hub-integration`  
**Date:** 2026-07-23  
**Status:** Investigation / spike

## Summary

NeMo AutoModel currently depends on several directly installed native kernel packages (`flash-attn`, `liger-kernel`, `mamba-ssm`, `transformer-engine`, `tilelang`, and others). Hugging Face's [`kernels`](https://huggingface.co/docs/kernels) library and the [`kernels-community`](https://huggingface.co/kernels-community) Hub organization provide pre-built, versioned, dynamically loaded replacements for many of these.

**Key finding:** AutoModel already pins `transformers==5.12.1`, which ships full Hub-kernel integration (`transformers.integrations.hub_kernels`). Partial Hub support is therefore already available for Hugging Face model paths when the `kernels` package is installed, but AutoModel's own kernel layer does not expose or fully leverage it yet.

The integration opportunity is to centralize kernel loading through the Hub instead of maintaining parallel install and compile paths, while keeping NVIDIA-specific backends (TE, DeepEP, and TileLang) on their existing paths.

---

## Kernels Ecosystem Background

| Component | Role |
|---|---|
| [`kernels`](https://github.com/huggingface/kernels) | Python loader: `get_kernel()`, `has_kernel()`, `kernelize()`, lockfiles |
| [`kernel-builder`](https://huggingface.co/docs/kernels) | Deterministic build and publish pipeline for Hub kernels |
| [`kernels-community`](https://huggingface.co/kernels-community) | Curated Hub kernels (flash-attn2/3/4, liger-kernels, mamba-ssm, activation, rotary, megablocks, and more) |

### Upstream API (from `huggingface/kernels`)

The loader lives in [`kernels/src/kernels/`](https://github.com/huggingface/kernels/tree/main/kernels/src/kernels). Key entry points:

| API | Purpose |
|---|---|
| `get_kernel(repo_id, version=1)` | Download and import a Hub kernel module (cached on disk) |
| `has_kernel(repo_id, version=1)` | Cheap compatibility check for current PyTorch/CUDA |
| `get_kernel_variants(...)` | Full resolution trace when `has_kernel` returns False |
| `kernelize(model, mode=Mode.TRAINING)` | Replace layer `forward` with Hub kernel implementations |
| `load_kernel` / `get_locked_kernel` | Offline or reproducible loads from `kernels lock` lockfiles |
| `LOCAL_KERNELS=repo=path` | Dev override that redirects a Hub id to a local `kernel-builder` build |

Usage pattern (from upstream [integration guide](https://github.com/huggingface/kernels/blob/main/kernel-builder/skills/cuda-kernels/references/huggingface-kernels-integration.md)):

```python
from kernels import get_kernel, has_kernel

if has_kernel("kernels-community/flash-attn2", version=1):
    fa2 = get_kernel("kernels-community/flash-attn2", version=1)
    out = fa2.flash_attn_func(q, k, v, causal=True)
    varlen = fa2.flash_attn_varlen_func  # same module, used by CP/packing paths
```

For container or offline builds, upstream recommends:

```bash
kernels lock kernels-community/flash-attn2 kernels-community/liger-kernels
kernels download
```

### AutoModel Wrapper (This Branch)

- `nemo_automodel/components/kernels/hub.py`: `get_hub_kernel()`, `has_hub_kernel()`, `get_flash_attn_varlen_func()`, `has_flash_attn_available()`
- `nemo_automodel/components/kernels/config.py`: `HubKernelConfig` for native `backend.attn="hub"`
- Unit tests: `tests/unit_tests/components/kernels/`

Blockdiag CP varlen uses `get_flash_attn_varlen_func()`. Remaining direct-import sites: KimiVL, BAGEL, EAGLE ring attention.

---

## Why Not Copy the QuACK Backend Pattern

QuACK adds `"quack"` to each `BackendConfig` field (`linear`, `rms_norm`, `rope`) and wires factory branches in `utils.py` plus `_apply_backend_module_overrides` in `model_init.py`. That fits pip optional deps where NeMo owns native model construction.

Hub kernels differ: Transformers already loads attention from Hub repo ids and replaces norms, MLP, and linear through `use_kernels=True`. Duplicating that in NeMo factories would fight Transformers.

| Concern | QuACK-Style on Every Field | Lean Integration |
|---|---|---|
| HF RMSNorm / MLP / Linear | Duplicate Hub loading | `use_kernels=True` |
| HF flash attention | Custom factory branch | `attn_implementation="kernels-community/flash-attn2"` |
| Native MLA attention | N/A | `backend.attn: hub` |
| Liger conflict | Separate pip path | `use_kernels=True` disables `_patch_liger_kernel` |

Example recipes (see `examples/llm_benchmark/qwen/`):

- `qwen2_5_7b_hub_kernels.yaml`: Hub flash-attn2 attention only
- `qwen2_5_7b_hub_kernels_layers.yaml`: Hub flash-attn2 plus `use_kernels` layer replacements

Install: `uv sync --extra hub_kernels`

`attn_implementation` and `use_kernels` are independent. A Hub repo id on `attn_implementation` selects the attention backend only. Add `use_kernels: true` (and `use_liger_kernel: false`) when non-attention layers should also come from Hub.

---

Transformers 5.x extends this with:

- `attn_implementation="kernels-community/flash-attn2"`: Hub attention backends
- `use_kernels=True`: auto-replace RMSNorm, MLP, Linear, activations, RoPE, causal LM loss, and related ops
- `KernelConfig`: per-layer and per-device kernel mapping
- **Automatic fallback:** `attn_implementation="flash_attention_2"` uses compiled `flash-attn` if present, else Hub `kernels-community/flash-attn2`

---

## Current AutoModel Kernel Architecture

### HF Wrapper Path (`nemo_automodel/_transformers/`)

| File | What It Does Today |
|---|---|
| `kernel_patches.py` | FA2/FA3/FA4 and Hub availability; fallback ladder; Liger and SDPA patching |
| `auto_model.py` | Passes `attn_implementation`; forwards `use_kernels` / `kernel_config`; skips Liger when `use_kernels=true` |

Default attention selection (`kernel_patches.py`):

```python
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2" if has_flash_attn_available() else "sdpa"
```

`has_flash_attn_available()` checks compiled `flash-attn` first, then Hub FA2/3/4.

### Native Model Path (`BackendConfig`)

NeMo-native models use `BackendConfig` (`components/models/common/utils.py`):

- `attn`: `te | sdpa | flex | eager | hub | tilelang`
- `hub_kernels`: optional repo settings when `attn="hub"`
- `linear`, `rms_norm`, `experts`, `dispatcher`, and related fields

HF `attn_implementation` and `use_kernels` apply to the transformers load path, not native factories.

### Direct `flash_attn` Imports (Bypass Transformers)

These call `flash_attn` APIs directly and would **not** benefit from transformers' Hub fallback:

| Location | Usage |
|---|---|
| `components/distributed/blockdiag_cp/kernels.py` | `get_flash_attn_varlen_func()` (Hub-aware) |
| `components/speculative/eagle/ring_attention.py` | Private `_flash_attn_forward/_backward` (pinned to FA 2.8.x ABI) |
| `components/speculative/eagle/draft_llama.py` | `flash_attn_func` / `flash_attn_varlen_func` |
| `components/models/kimivl/model.py` | `flash_attn_varlen_func` |
| `components/models/kimi_k25_vl/model.py` | `flash_attn_varlen_func` |
| `components/models/kimi_k3/vision.py` | `flash_attn_varlen_func` (new in upstream) |
| `components/models/bagel/modeling_qwen2_packed.py` | `flash_attn_varlen_func` for packed inference |
| `components/models/bagel/modeling_siglip_navit.py` | `flash_attn_varlen_func` |
| `components/models/llava_onevision/rice_vit.py` | `flash_attn_varlen_func` |

### Dependencies (`pyproject.toml`)

| Optional Group | Packages |
|---|---|
| `fa` | `flash-attn<=2.8.3` |
| `cuda` | `mamba-ssm`, `causal-conv1d`, `transformer-engine`, `tilelang`, and related |
| `diffusion_kernels` | `kernels` |
| `hub_kernels` | `kernels>=0.11.0` |
| (implicit) | `liger-kernel` through runtime import in `kernel_patches.py` |

Docker images compile `flash-attn` from source. This is a major install-time cost Hub kernels could eliminate.

---

## What Already Works

With `uv sync --extra hub_kernels`:

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-3B
  attn_implementation: kernels-community/flash-attn2
```

`flash_attention_2` also works: Transformers uses pip `flash-attn` when installed, else Hub FA2.

For Hub layer replacements (RMSNorm, MLP, Linear, etc.), see `qwen2_5_7b_hub_kernels_layers.yaml`:

```yaml
  attn_implementation: kernels-community/flash-attn2
  use_kernels: true
  use_liger_kernel: false
```

Implemented on this branch: Hub-aware `kernel_patches`; `use_kernels` passthrough; Liger gated when `use_kernels=true`; blockdiag CP varlen through Hub.

Remaining gaps:

1. Several models still import pip `flash_attn` directly (KimiVL, Kimi K2.5/K3 vision,
   BAGEL, LLaVA-OneVision Rice ViT, EAGLE draft/ring attention).
2. Native models need `backend.attn: hub`; HF `attn_implementation` does not apply there.
3. TE and Hub FA remain mutually exclusive (TE pins compiled `flash-attn`).
4. Scoped partial CUDA graphs (`backend.cuda_graph`) require `attn="te"`; incompatible
   with `attn="hub"`.

---

## Hub Kernels and AutoModel Feature Map

| AutoModel Feature | Current Backend | Hub Kernel Available? | Notes |
|---|---|---|---|
| HF attention (FA2/3/4) | `flash-attn` pip or Hub | ✅ `kernels-community/flash-attn{2,3,4}` | Transformers dispatch; AutoModel fallback ladder is Hub-aware |
| Liger (RMSNorm, MLP, Linear, loss) | `liger_kernel` pip | ✅ `kernels-community/liger-kernels` | `use_kernels=True` replaces `_patch_liger_kernel` |
| Mamba / GDN conv | `mamba-ssm`, `causal-conv1d` | ✅ `kernels-community/mamba-ssm` | Includes `causal_conv1d_*`, selective scan, Mamba2 |
| Activations (GELU, SiLU) | PyTorch or TE | ✅ `kernels-community/activation` | Inference and compile modes |
| RoPE | TE fused or torch | ✅ `kernels-community/rotary` | TE fused RoPE currently force-disabled (#3027) |
| MoE expert GEMM | TE, grouped_gemm, or torch_mm | ⚠️ `kernels-community/megablocks` | Different API; not a drop-in for DeepEP path |
| CP blockdiag varlen | `get_flash_attn_varlen_func()` | ✅ through Hub FA2 module | Wired on this branch |
| EAGLE ring attention | Direct FA 2.8.x private API | ❓ | Uses `_flash_attn_forward` positional ABI; Hub FA2 might differ and needs a parity test |
| TE attention and FP8 | `transformer-engine` | ❌ | NVIDIA proprietary; stays as direct dep |
| DeepEP or UCCL-EP | `deep_ep` | ❌ | Not in kernels-community |
| TileLang (DSV4, GLM-DSA) | `tilelang`, `tile-kernels` | ❌ | Custom NVIDIA/vendor kernels |
| FLA (linear attention) | `flash-linear-attention` | ✅ `kernels-community/fla` | Hub repo exists; not in transformers default map yet |
| FlexAttn or MagiAttention | PyTorch or custom | ❌ | Custom CP dispatch |
| Quantization (BNB/GPTQ) | `bitsandbytes` pip | ⚠️ `kernels-community/quantization-*` | CPU-focused Hub builds; evaluate for QLoRA |
| FP8 GEMM | TE or torchao | ⚠️ `kernels-community/finegrained-fp8`, `deep-gemm` | Experimental; not TE-compatible |

---

## Full `kernels-community` Catalog (54 Kernels)

Source: [kernels-community/kernels](https://huggingface.co/kernels-community/kernels) using `GET https://huggingface.co/api/kernels?author=kernels-community` (July 24, 2026). Source repos live in [huggingface/kernels-community](https://github.com/huggingface/kernels-community).

Legend: **P0** = high-priority AutoModel integration candidate · **P1** = useful secondary · **P2** = niche or future · **n/a** = no direct AutoModel path today

### Attention and Sequence Ops

| Hub Repo | Drivers | AutoModel | Notes |
|---|---|---|---|
| [`flash-attn2`](https://huggingface.co/kernels-community/flash-attn2) | cuda,xpu,cpu | **P0** | `flash_attn_func`, `flash_attn_varlen_func`, packed/KV APIs. Replaces `fa` pip. |
| [`flash-attn3`](https://huggingface.co/kernels-community/flash-attn3) | cuda | **P0** | Hopper FA3; also `vllm-flash-attn3`, `sgl-flash-attn3` variants |
| [`flash-attn4`](https://huggingface.co/kernels-community/flash-attn4) | cuda | **P1** | Beta FA4; conflicts with tilelang ffi pin in current container |
| [`flash-attn-ops`](https://huggingface.co/kernels-community/flash-attn-ops) | cuda,rocm,xpu | **P1** | `cross_entropy_loss`, `rms_norm_fn`, `apply_rotary`. Auxiliary FA ecosystem ops. |
| [`flash-mla`](https://huggingface.co/kernels-community/flash-mla) | cuda | **P2** | Multi-latent attention (DeepSeek-style MLA kernels) |
| [`sage-attention`](https://huggingface.co/kernels-community/sage-attention) | cuda | **P2** | Approximate attention |
| [`paged-attention`](https://huggingface.co/kernels-community/paged-attention) | cuda,rocm,metal | **P2** | Inference paging; vLLM-style serving |
| [`vllm-flash-attn3`](https://huggingface.co/kernels-community/vllm-flash-attn3) | cuda | **P2** | vLLM-flavored FA3 build |
| [`sgl-flash-attn3`](https://huggingface.co/kernels-community/sgl-flash-attn3) | cuda | **P2** | SGLang-flavored FA3 and varlen |
| [`aiter-flash-attn`](https://huggingface.co/kernels-community/aiter-flash-attn) | rocm | **P2** | AMD AITER flash attention |
| [`aiter-flash-attn-ck`](https://huggingface.co/kernels-community/aiter-flash-attn-ck) | rocm | **P2** | AMD composable-kernel FA |
| [`metal-flash-sdpa`](https://huggingface.co/kernels-community/metal-flash-sdpa) | metal | **P2** | Apple Metal SDPA |
| [`msa`](https://huggingface.co/kernels-community/msa) | cuda | **P2** | Block-sparse attention (`sparse_atten_func`); transformers has dedicated wrapper |
| [`mra`](https://huggingface.co/kernels-community/mra) | cuda | **P2** | Multi-resolution attention |
| [`yoso`](https://huggingface.co/kernels-community/yoso) | cuda | **P2** | Efficient attention variant |

### Normalization, Activations, and RoPE

| Hub Repo | Drivers | AutoModel | Notes |
|---|---|---|---|
| [`liger-kernels`](https://huggingface.co/kernels-community/liger-kernels) | cuda,rocm,xpu | **P0** | `LigerRMSNorm`, SwiGLU/GEGLU MLP, `LigerLinear`, causal LM loss. Replaces `_patch_liger_kernel`. |
| [`activation`](https://huggingface.co/kernels-community/activation) | cuda,metal,cpu | **P1** | `silu_and_mul`, `gelu_and_mul`, GELU variants. Transformers default for activations. |
| [`rotary`](https://huggingface.co/kernels-community/rotary) | cuda,xpu,cpu | **P1** | `apply_rotary_transformers`. Transformers default RoPE on CUDA/XPU. |
| [`aiter-rope`](https://huggingface.co/kernels-community/aiter-rope) | rocm | **P2** | AMD RoPE |
| [`layer-norm`](https://huggingface.co/kernels-community/layer-norm) | cuda | **P1** | LayerNorm/RMSNorm Triton builds |
| [`rmsnorm`](https://huggingface.co/kernels-community/rmsnorm) | xpu,cpu | **P2** | Intel XPU RMSNorm (transformers default on XPU) |
| [`tinygrad-rms`](https://huggingface.co/kernels-community/tinygrad-rms) | cuda | **P2** | Alternative RMSNorm |
| [`relu`](https://huggingface.co/kernels-community/relu) | all | **P2** | Cross-platform ReLU (highest download count, generic) |

### SSM, Linear Attention, and RWKV

| Hub Repo | Drivers | AutoModel | Notes |
|---|---|---|---|
| [`mamba-ssm`](https://huggingface.co/kernels-community/mamba-ssm) | cuda | **P0** | `selective_scan_fn`, `causal_conv1d_*`, `Mamba2`. Replaces `mamba-ssm` and `causal-conv1d` pip. |
| [`causal-conv1d`](https://huggingface.co/kernels-community/causal-conv1d) | cuda | **P1** | Standalone conv1d (also bundled in `mamba-ssm`) |
| [`fla`](https://huggingface.co/kernels-community/fla) | cuda | **P1** | Flash-linear-attention ops. Replaces `flash-linear-attention` pip for GDN and linear-attn models. |
| [`rwkv`](https://huggingface.co/kernels-community/rwkv) | cuda | **P2** | RWKV time-mix kernels |

### MoE and GEMM

| Hub Repo | Drivers | AutoModel | Notes |
|---|---|---|---|
| [`megablocks`](https://huggingface.co/kernels-community/megablocks) | cuda,rocm,xpu,cpu | **P1** | `MegaBlocksMoeMLP`, dropless MoE. Transformers default MoE MLP; **not** DeepEP-compatible. |
| [`megablocks-rocm`](https://huggingface.co/kernels-community/megablocks-rocm) | rocm | **P2** | ROCm megablocks variant |
| [`triton-kernels`](https://huggingface.co/kernels-community/triton-kernels) | cuda | **P1** | Triton MoE routing/SwiGLU (`swiglu`, `routing`). Supports mxfp4/bf16. |
| [`triton-moe`](https://huggingface.co/kernels-community/triton-moe) | — | **P2** | MoE-specific Triton ops |
| [`scattermoe`](https://huggingface.co/kernels-community/scattermoe) | cuda,rocm,xpu | **P2** | Scatter MoE kernels |
| [`sonic-moe`](https://huggingface.co/kernels-community/sonic-moe) | cuda | **P2** | `KernelBackendMoE`, routing kernels |
| [`vllm-moe`](https://huggingface.co/kernels-community/vllm-moe) | cuda | **P2** | vLLM MoE kernels |
| [`deep-gemm`](https://huggingface.co/kernels-community/deep-gemm) | cuda | **P2** | DeepSeek GEMM kernels |
| [`gemm`](https://huggingface.co/kernels-community/gemm) | — | **P2** | Generic GEMM |
| [`triton-scaled-mm`](https://huggingface.co/kernels-community/triton-scaled-mm) | — | **P2** | Scaled matmul (quant) |

### Quantization and FP8

| Hub Repo | Drivers | AutoModel | Notes |
|---|---|---|---|
| [`quantization-bitsandbytes`](https://huggingface.co/kernels-community/quantization-bitsandbytes) | cpu | **P1** | BNB quant kernels. Evaluate as a replacement for pip `bitsandbytes` for QLoRA. |
| [`quantization-gptq`](https://huggingface.co/kernels-community/quantization-gptq) | cpu | **P2** | GPTQ CPU kernels |
| [`quantization-eetq`](https://huggingface.co/kernels-community/quantization-eetq) | cuda | **P2** | EETQ quant |
| [`finegrained-fp8`](https://huggingface.co/kernels-community/finegrained-fp8) | cuda,rocm,xpu | **P2** | FP8 quant kernels |
| [`fp8-fbgemm`](https://huggingface.co/kernels-community/fp8-fbgemm) | cuda,rocm,xpu | **P2** | FBGEMM FP8 |

### Vision, Diffusion, and Domain-Specific

| Hub Repo | Drivers | AutoModel | Notes |
|---|---|---|---|
| [`deformable-detr`](https://huggingface.co/kernels-community/deformable-detr) | cuda | **P2** | Deformable attention. Transformers default for DETR-style models. |
| [`cv-utils`](https://huggingface.co/kernels-community/cv-utils) | cuda | **P2** | CV helper kernels |
| [`trimul-gpumode`](https://huggingface.co/kernels-community/trimul-gpumode) | cuda,rocm,xpu | **P2** | AlphaFold TriMul |
| [`punica-sgmv`](https://huggingface.co/kernels-community/punica-sgmv) | cuda | **P2** | Multi-LoRA SGMV (serving) |

### Platform-Specific (Metal, ROCm AITER, and GPT-OSS)

| Hub Repo | Drivers | AutoModel | Notes |
|---|---|---|---|
| [`gpt-oss-triton-kernels`](https://huggingface.co/kernels-community/gpt-oss-triton-kernels) | cuda,rocm,xpu | **P1** | GPT-OSS model-specific Triton kernels |
| [`gpt-oss-metal-kernels`](https://huggingface.co/kernels-community/gpt-oss-metal-kernels) | metal | **P2** | GPT-OSS on Apple Metal |
| [`mlx-rmsnorm`](https://huggingface.co/kernels-community/mlx-rmsnorm) | metal | **P2** | MLX RMSNorm |
| [`mlx-quantization-metal-kernels`](https://huggingface.co/kernels-community/mlx-quantization-metal-kernels) | metal | **P2** | MLX quant |
| [`bitsandbytes-mps`](https://huggingface.co/kernels-community/bitsandbytes-mps) | metal | **P2** | BNB on MPS |
| [`aiter-kernels`](https://huggingface.co/kernels-community/aiter-kernels) | rocm | **P2** | AMD AITER kernel bundle |

### Optimizer and Miscellaneous

| Hub Repo | Drivers | AutoModel | Notes |
|---|---|---|---|
| [`adam-atan2`](https://huggingface.co/kernels-community/adam-atan2) | — | **P2** | Optimizer kernel |

### Suggested `kernels lock` Set for AutoModel Containers

Minimum set covering HF training path without compiled `flash-attn` or `liger-kernel`:

```bash
kernels lock \
  kernels-community/flash-attn2 \
  kernels-community/liger-kernels \
  kernels-community/activation \
  kernels-community/rotary \
  kernels-community/mamba-ssm
kernels download
```

Extended set for MoE, linear-attention, and native-model experiments:

```bash
kernels lock \
  kernels-community/flash-attn2 \
  kernels-community/flash-attn3 \
  kernels-community/liger-kernels \
  kernels-community/megablocks \
  kernels-community/mamba-ssm \
  kernels-community/fla \
  kernels-community/triton-kernels
kernels download
```

---

## Proposed Integration Plan

### Phase 0 Passthrough Validation (One to Two Days)

**Goal:** Confirm Hub kernels work end-to-end through existing `NeMoAutoModel` without refactors.

- [ ] Install `kernels` in the dev container
- [ ] Run a minimal finetune with `attn_implementation=kernels-community/flash-attn2` (no `flash-attn` pip)
- [ ] Run the same with `attn_implementation=flash_attention_2` and no compiled FA. Verify transformers Hub fallback.
- [ ] Document working recipe YAML

### Phase 1 Hub-Aware Availability and Passthrough (Small PR)

**Goal:** Make AutoModel's kernel layer Hub-aware without breaking existing installs.

1. **Add optional dependency group** (extend or alias `fa`):

   ```toml
   hub_kernels = ["kernels>=0.11.0"]
   fa = ["flash-attn<=2.8.3"]  # keep for TE-compat / ring-attn ABI pinning
   fa_or_hub = ["nemo_automodel[hub_kernels]"]  # Hub-only path
   ```

2. **Use `nemo_automodel/components/kernels/hub.py`** (added on this branch). This module is a thin, cached wrapper around upstream `kernels.get_kernel` and `has_kernel`.

3. **Update `kernel_patches.py`:**
   - Extend `has_flash_attn()` to check Hub via `has_kernel()` or transformers `is_kernels_available()`
   - Teach `_apply_preload_overrides` that `kernels-community/flash-attn*` counts as flash for packed sequences
   - Extend `_get_next_fallback_attn` to handle Hub repo IDs (or delegate to transformers' resolver)

4. **Plumb `use_kernels` and `kernel_config`** through `NeMoAutoModel.from_pretrained` kwargs (pass-through, no new API surface beyond forwarding).

5. **Gate Liger:** when `use_kernels=True`, skip `_patch_liger_kernel` (transformers handles it).

### Phase 2 Replace Direct `flash_attn` Imports (Medium PR)

**Goal:** CP, speculative, and VLM paths load varlen kernels from the Hub when the pip package is absent.

| File | Change |
|---|---|
| `blockdiag_cp/kernels.py` | Done. Uses `get_flash_attn_varlen_func()`. |
| `kimivl/model.py`, `kimi_k25_vl/model.py`, `kimi_k3/vision.py` | Same helper |
| `bagel/modeling_qwen2_packed.py`, `bagel/modeling_siglip_navit.py` | Same helper |
| `llava_onevision/rice_vit.py` | Same helper |
| `eagle/draft_llama.py`, `eagle/ring_attention.py` | **High risk.** Private FA ABI; might keep pip pin. |

### Phase 3 Liger Consolidation (Medium PR)

 Default `use_kernels=True` for HF models when `kernels` is installed
- Deprecate direct `liger_kernel` import path
- Add `kernels lock` to CI/container for reproducible builds

### Phase 4 Native `BackendConfig` (Optional, Larger Scope)

Extend `BackendConfig.attn` with Hub-backed options or a separate `hub_kernels: KernelConfig` field for NeMo-native model layers. This is a larger design decision. Native models currently bypass transformers' attention dispatch entirely.

---

## Container and CI Implications

| Today | With Hub Kernels |
|---|---|
| Docker builds compile `flash-attn` from source (slow, fragile) | `pip install kernels` plus `kernels lock` or `kernels download` at build time |
| `fa` optional extra for users | `hub_kernels` extra; `fa` kept for TE version pinning |
| Per-CUDA wheel matrix in CI | Hub resolves compatible build variant at runtime |
| Air-gapped clusters | Pre-download with `kernels download` plus lockfile in image |

Recommended container change:

```dockerfile
RUN uv pip install "kernels>=0.11.0" \
 && kernels lock kernels-community/flash-attn2 kernels-community/liger-kernels \
 && kernels download  # bake into image for offline use
```

---

## Risks and Blockers

| Risk | Severity | Mitigation |
|---|---|---|
| EAGLE ring attention pins FA 2.8.x private API | **High** | Keep `fa` dep for speculative; Hub path for standard HF attention only |
| TE requires specific `flash-attn` version | **High** | Document mutual exclusion; TE path keeps compiled FA |
| FSDP2 with Hub kernel autograd | **Medium** | Parity tests on 1-GPU and multi-GPU finetune |
| Hub download at training start (latency) | **Low** | `kernels lock` plus pre-download in container |
| `USE_HUB_KERNELS=0` env disables all Hub loading | **Low** | Document; default YES in transformers |
| Packed sequence with CP override forces SDPA | **Medium** | Existing limitation; Hub FA varlen might enable CP and packing later |
| tilelang, FA4, and apache-tvm-ffi conflict | **High** | Unrelated to Hub; existing pyproject constraint remains |

---

## Recommended Next Steps

1. **Phase 0 spike:** run one L0 finetune on branch with `kernels` installed, no `flash-attn` pip, `attn_implementation=kernels-community/flash-attn2`.
2. **Phase 1 PR:** hub-aware availability checks and kwargs passthrough (minimal, backward compatible).
3. **Decision point:** after EAGLE ring-attn parity test, decide whether speculative stays on compiled FA permanently.
4. **Container experiment:** build image without `flash-attn` source compile, Hub-only FA2; measure build time and runtime parity.

---

## References

- [huggingface/kernels repo](https://github.com/huggingface/kernels): loader source, CLI, lockfiles
- [Upstream integration guide](https://github.com/huggingface/kernels/blob/main/kernel-builder/skills/cuda-kernels/references/huggingface-kernels-integration.md)
- [Upstream example script](https://github.com/huggingface/kernels/blob/main/kernel-builder/skills/cuda-kernels/scripts/huggingface_kernels_example.py)
- [Kernels Quickstart](https://huggingface.co/docs/kernels/en/basic-usage)
- [Transformers Loading Kernels](https://huggingface.co/docs/transformers/main/kernel_doc/loading_kernels)
- [Transformers `hub_kernels.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py)
- [kernels-community catalog (54 kernels)](https://huggingface.co/kernels-community/kernels)
- [kernels-community source repo](https://github.com/huggingface/kernels-community)
- AutoModel: `nemo_automodel/components/kernels/hub.py`, `nemo_automodel/_transformers/kernel_patches.py`, `nemo_automodel/components/models/common/utils.py` (`BackendConfig`)
