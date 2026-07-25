# Kernels Hub Integration Investigation

**Branch:** `kashif/investigate/kernels-hub-integration`  
**Date:** 2026-07-23  
**Status:** Investigation / spike

## Summary

NeMo AutoModel currently depends on several **directly installed** native kernel packages (`flash-attn`, `liger-kernel`, `mamba-ssm`, `transformer-engine`, `tilelang`, etc.). Hugging Face's [`kernels`](https://huggingface.co/docs/kernels) library and the [`kernels-community`](https://huggingface.co/kernels-community) Hub organization provide **pre-built, versioned, dynamically loaded** replacements for many of these.

**Key finding:** AutoModel already pins `transformers==5.12.1`, which ships full Hub-kernel integration (`transformers.integrations.hub_kernels`). Partial Hub support is therefore **already available** for HF-model paths when the `kernels` package is installed — but AutoModel's own kernel layer does not expose or fully leverage it yet.

The integration opportunity is to **centralize kernel loading through the Hub** instead of maintaining parallel install/compile paths, while keeping NVIDIA-specific backends (TE, DeepEP, TileLang) on their existing paths.

---

## Background: The Kernels Ecosystem

| Component | Role |
|---|---|
| [`kernels`](https://github.com/huggingface/kernels) | Python loader — `get_kernel()`, `has_kernel()`, `kernelize()`, lockfiles |
| [`kernel-builder`](https://huggingface.co/docs/kernels) | Deterministic build/publish pipeline for Hub kernels |
| [`kernels-community`](https://huggingface.co/kernels-community) | Curated Hub kernels (flash-attn2/3/4, liger-kernels, mamba-ssm, activation, rotary, megablocks, …) |

### Upstream API (from `huggingface/kernels`)

The loader lives in [`kernels/src/kernels/`](https://github.com/huggingface/kernels/tree/main/kernels/src/kernels). Key entry points:

| API | Purpose |
|---|---|
| `get_kernel(repo_id, version=1)` | Download + import a Hub kernel module (cached on disk) |
| `has_kernel(repo_id, version=1)` | Cheap compatibility check for current PyTorch/CUDA |
| `get_kernel_variants(...)` | Full resolution trace when `has_kernel` returns False |
| `kernelize(model, mode=Mode.TRAINING)` | Replace layer `forward` with Hub kernel implementations |
| `load_kernel` / `get_locked_kernel` | Offline/reproducible loads from `kernels lock` lockfiles |
| `LOCAL_KERNELS=repo=path` | Dev override — redirect Hub id to a local `kernel-builder` build |

Usage pattern (from upstream [integration guide](https://github.com/huggingface/kernels/blob/main/kernel-builder/skills/cuda-kernels/references/huggingface-kernels-integration.md)):

```python
from kernels import get_kernel, has_kernel

if has_kernel("kernels-community/flash-attn2", version=1):
    fa2 = get_kernel("kernels-community/flash-attn2", version=1)
    out = fa2.flash_attn_func(q, k, v, causal=True)
    varlen = fa2.flash_attn_varlen_func  # same module, used by CP/packing paths
```

For container/offline builds, upstream recommends:

```bash
kernels lock kernels-community/flash-attn2 kernels-community/liger-kernels
kernels download
```

### AutoModel wrapper (this branch)

A thin spike module wraps the upstream API without replacing existing call sites yet:

- `nemo_automodel/components/kernels/hub.py` — `get_hub_kernel()`, `has_hub_kernel()`, `get_flash_attn_varlen_func()`, `has_flash_attn_available()`
- Unit tests: `tests/unit_tests/components/kernels/test_hub.py`

Phase 2 migrated blockdiag CP varlen to `get_flash_attn_varlen_func()`. Remaining direct-import sites: KimiVL, BAGEL, EAGLE ring attention.

---

## Design: Why Not Copy the QuACK Backend Pattern?

QuACK integration adds `"quack"` to **every** `BackendConfig` field (`linear`, `rms_norm`, `rope`) and wires factory branches in `utils.py` plus `_apply_backend_module_overrides` in `model_init.py`. That pattern fits **pip-installed optional deps** where NeMo owns native model construction.

Hub kernels are different:

| Concern | QuACK-style `"kernels"` on each field | Lean integration (implemented) |
|---|---|---|
| HF-model RMSNorm/MLP/Linear | Duplicate Hub loading in NeMo factories | `use_kernels=True` → Transformers `kernelize` |
| HF flash attention | Custom factory branch | `attn_implementation="kernels-community/flash-attn2"` (Transformers dispatch) |
| Native MLA attention | N/A in QuACK | `backend.attn: hub` + `initialize_attn_module_and_func` |
| Liger conflict | Separate pip path | `use_kernels=True` disables `_patch_liger_kernel` |
| Config surface | Sprawl across 5+ backend fields | `use_kernels`, `kernel_config`, optional `HubKernelConfig` |

**Rule of thumb:** use Transformers passthrough for anything `hub_kernels.kernelize` already replaces; use NeMo helpers only where Transformers does not run (blockdiag varlen, native MLA `attn: hub`).

Example recipes:

- `examples/llm_benchmark/qwen/qwen2_5_7b_hub_kernels.yaml` — Hub FA2 attention only (no `use_kernels`)
- `examples/llm_benchmark/qwen/qwen2_5_7b_hub_kernels_layers.yaml` — Hub FA2 + Hub layer kernels (`use_kernels=true`)

Optional extra: `uv sync --extra hub_kernels`

### Attention vs layer kernels (orthogonal flags)

Transformers treats these as **separate** code paths:

| Setting | Scope | Required when |
|---|---|---|
| `attn_implementation: kernels-community/flash-attn2` | Attention only (`load_and_register_attn_kernel`) | You want Hub flash attention |
| `attn_implementation: flash_attention_2` | Attention; falls back to Hub FA2 when pip `flash-attn` absent | Shorthand for Hub-or-pip FA2 |
| `use_kernels: true` | RMSNorm, MLP, Linear, activations, RoPE, fused CE, … (`kernelize`) | You want Hub replacements for **non-attention** layers |
| `use_liger_kernel: true` (default) | Same layer types via **pip** `liger_kernel` | Default AutoModel path; disable when `use_kernels=true` |

**You do not need `use_kernels` when `attn_implementation` is already a Hub repo id** — unless you also want Hub layer kernels instead of torch or pip Liger.

```yaml
# Attention only — sufficient for Hub FA2
attn_implementation: kernels-community/flash-attn2

# Full Hub stack — both flags, disable pip Liger to avoid double patch
attn_implementation: kernels-community/flash-attn2
use_kernels: true
use_liger_kernel: false
```

---

Transformers 5.x extends this with:

- `attn_implementation="kernels-community/flash-attn2"` — Hub attention backends
- `use_kernels=True` — auto-replace RMSNorm, MLP, Linear, activations, RoPE, causal LM loss, etc.
- `KernelConfig` — per-layer / per-device kernel mapping
- **Automatic fallback:** `attn_implementation="flash_attention_2"` uses compiled `flash-attn` if present, else Hub `kernels-community/flash-attn2`

---

## Current AutoModel Kernel Architecture

### 1. HF wrapper path (`nemo_automodel/_transformers/`)

| File | What it does today |
|---|---|
| `kernel_patches.py` | Direct `safe_import("flash_attn")`; FA2/FA3/FA4 detection; fallback ladder; Liger patching via `liger_kernel.transformers`; SDPA patching |
| `auto_model.py` | Passes `attn_implementation`; forwards `use_kernels` / `kernel_config`; skips Liger when `use_kernels=true` |

Default attention selection (`kernel_patches.py`):

```python
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2" if HAS_FA else "sdpa"
```

Availability is gated on **compiled package presence**, not Hub kernel availability.

### 2. Native model path (`BackendConfig`)

NeMo-native models use `BackendConfig` (`components/models/common/utils.py`) with explicit backends:

- `attn`: `te | sdpa | flex | eager | tilelang`
- `linear`, `rms_norm`, `experts`, `dispatcher`, …

This path is **orthogonal** to the HF Hub-kernel system and does not use `kernels` today.

### 3. Direct `flash_attn` imports (bypass transformers)

These call `flash_attn` APIs directly and would **not** benefit from transformers' Hub fallback:

| Location | Usage |
|---|---|
| `components/distributed/blockdiag_cp/kernels.py` | `flash_attn_varlen_func` for CP varlen attention |
| `components/speculative/eagle/ring_attention.py` | Private `_flash_attn_forward/_backward` (pinned to FA 2.8.x ABI) |
| `components/models/kimivl/model.py` | `flash_attn_varlen_func` |
| `components/models/bagel/modeling_qwen2_packed.py` | `flash_attn_varlen_func` for packed inference |

### 4. Dependencies (`pyproject.toml`)

| Optional group | Packages |
|---|---|
| `fa` | `flash-attn<=2.8.3` |
| `cuda` | `mamba-ssm`, `causal-conv1d`, `transformer-engine`, `tilelang`, … |
| `diffusion_kernels` | `kernels` (declared but **no Python imports yet**) |
| (implicit) | `liger-kernel` via runtime import in `kernel_patches.py` |

Docker images compile `flash-attn` from source; this is a major install-time cost Hub kernels could eliminate.

---

## What Already Works

With `uv sync --extra hub_kernels`:

```yaml
# Hub flash attention only — use_kernels not required
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-3B
  attn_implementation: kernels-community/flash-attn2

# Or shorthand: pip flash-attn when installed, else Hub FA2
  attn_implementation: flash_attention_2

# Optional: also Hub-ize norms / MLP / linear (see qwen2_5_7b_hub_kernels_layers.yaml)
  use_kernels: true
  use_liger_kernel: false
```

**Fixed on this branch:** Hub-aware `kernel_patches` (availability, packed sequences, fallback ladder); `use_kernels` / `kernel_config` passthrough; Liger gated off when `use_kernels=true`; blockdiag CP varlen via Hub.

**Remaining gaps:**

1. KimiVL / BAGEL / EAGLE ring attention still import pip `flash_attn` directly.
2. Native models need `backend.attn: hub` for MLA factories; HF `attn_implementation` does not apply there.
3. TE + Hub FA mutual exclusion still applies (TE pins compiled `flash-attn`).

---

## Hub Kernels ↔ AutoModel Feature Map

| AutoModel feature | Current backend | Hub kernel available? | Notes |
|---|---|---|---|
| HF attention (FA2/3/4) | `flash-attn` pip package | ✅ `kernels-community/flash-attn{2,3,4}` | Transformers handles wrapper; AutoModel needs availability + fallback fixes |
| Liger (RMSNorm, MLP, Linear, loss) | `liger_kernel` pip | ✅ `kernels-community/liger-kernels` | Transformers `use_kernels=True` replaces `_patch_liger_kernel` |
| Mamba / GDN conv | `mamba-ssm`, `causal-conv1d` | ✅ `kernels-community/mamba-ssm` | Includes `causal_conv1d_*`, selective scan, Mamba2 |
| Activations (GELU, SiLU) | PyTorch / TE | ✅ `kernels-community/activation` | Inference + compile modes |
| RoPE | TE fused / torch | ✅ `kernels-community/rotary` | TE fused RoPE currently force-disabled (#3027) |
| MoE expert GEMM | TE / grouped_gemm / torch_mm | ⚠️ `kernels-community/megablocks` | Different API; not a drop-in for DeepEP path |
| CP blockdiag varlen | Direct `flash_attn_varlen_func` | ✅ via Hub FA2 module | Needs thin loader wrapper |
| EAGLE ring attention | Direct FA 2.8.x private API | ❓ | Uses `_flash_attn_forward` positional ABI; Hub FA2 may differ — needs parity test |
| TE attention / FP8 | `transformer-engine` | ❌ | NVIDIA proprietary; stays as direct dep |
| DeepEP / UCCL-EP | `deep_ep` | ❌ | Not in kernels-community |
| TileLang (DSV4, GLM-DSA) | `tilelang`, `tile-kernels` | ❌ | Custom NVIDIA/vendor kernels |
| FLA (linear attention) | `flash-linear-attention` | ✅ `kernels-community/fla` | Hub repo exists; not in transformers default map yet |
| FlexAttn / MagiAttention | PyTorch / custom | ❌ | Custom CP dispatch |
| Quantization (BNB/GPTQ) | `bitsandbytes` pip | ⚠️ `kernels-community/quantization-*` | CPU-focused Hub builds; evaluate for QLoRA |
| FP8 GEMM | TE / torchao | ⚠️ `kernels-community/finegrained-fp8`, `deep-gemm` | Experimental; not TE-compatible |

---

## Full `kernels-community` Catalog (54 kernels)

Source: [kernels-community/kernels](https://huggingface.co/kernels-community/kernels) via `GET https://huggingface.co/api/kernels?author=kernels-community` (2026-07-24). Source repos live in [huggingface/kernels-community](https://github.com/huggingface/kernels-community).

Legend: **P0** = high-priority AutoModel integration candidate · **P1** = useful secondary · **P2** = niche / future · **—** = no direct AutoModel path today

### Attention & sequence ops

| Hub repo | Drivers | AutoModel | Notes |
|---|---|---|---|
| [`flash-attn2`](https://huggingface.co/kernels-community/flash-attn2) | cuda,xpu,cpu | **P0** | `flash_attn_func`, `flash_attn_varlen_func`, packed/KV APIs — replaces `fa` pip |
| [`flash-attn3`](https://huggingface.co/kernels-community/flash-attn3) | cuda | **P0** | Hopper FA3; also `vllm-flash-attn3`, `sgl-flash-attn3` variants |
| [`flash-attn4`](https://huggingface.co/kernels-community/flash-attn4) | cuda | **P1** | Beta FA4; conflicts with tilelang ffi pin in current container |
| [`flash-attn-ops`](https://huggingface.co/kernels-community/flash-attn-ops) | cuda,rocm,xpu | **P1** | `cross_entropy_loss`, `rms_norm_fn`, `apply_rotary` — auxiliary FA ecosystem ops |
| [`flash-mla`](https://huggingface.co/kernels-community/flash-mla) | cuda | **P2** | Multi-latent attention (DeepSeek-style MLA kernels) |
| [`sage-attention`](https://huggingface.co/kernels-community/sage-attention) | cuda | **P2** | Approximate attention |
| [`paged-attention`](https://huggingface.co/kernels-community/paged-attention) | cuda,rocm,metal | **P2** | Inference paging; vLLM-style serving |
| [`vllm-flash-attn3`](https://huggingface.co/kernels-community/vllm-flash-attn3) | cuda | **P2** | vLLM-flavored FA3 build |
| [`sgl-flash-attn3`](https://huggingface.co/kernels-community/sgl-flash-attn3) | cuda | **P2** | SGLang-flavored FA3 + varlen |
| [`aiter-flash-attn`](https://huggingface.co/kernels-community/aiter-flash-attn) | rocm | **P2** | AMD AITER flash attention |
| [`aiter-flash-attn-ck`](https://huggingface.co/kernels-community/aiter-flash-attn-ck) | rocm | **P2** | AMD composable-kernel FA |
| [`metal-flash-sdpa`](https://huggingface.co/kernels-community/metal-flash-sdpa) | metal | **P2** | Apple Metal SDPA |
| [`msa`](https://huggingface.co/kernels-community/msa) | cuda | **P2** | Block-sparse attention (`sparse_atten_func`); transformers has dedicated wrapper |
| [`mra`](https://huggingface.co/kernels-community/mra) | cuda | **P2** | Multi-resolution attention |
| [`yoso`](https://huggingface.co/kernels-community/yoso) | cuda | **P2** | Efficient attention variant |

### Normalization, activations, RoPE

| Hub repo | Drivers | AutoModel | Notes |
|---|---|---|---|
| [`liger-kernels`](https://huggingface.co/kernels-community/liger-kernels) | cuda,rocm,xpu | **P0** | `LigerRMSNorm`, SwiGLU/GEGLU MLP, `LigerLinear`, causal LM loss — replaces `_patch_liger_kernel` |
| [`activation`](https://huggingface.co/kernels-community/activation) | cuda,metal,cpu | **P1** | `silu_and_mul`, `gelu_and_mul`, GELU variants — transformers default for activations |
| [`rotary`](https://huggingface.co/kernels-community/rotary) | cuda,xpu,cpu | **P1** | `apply_rotary_transformers` — transformers default RoPE on CUDA/XPU |
| [`aiter-rope`](https://huggingface.co/kernels-community/aiter-rope) | rocm | **P2** | AMD RoPE |
| [`layer-norm`](https://huggingface.co/kernels-community/layer-norm) | cuda | **P1** | LayerNorm/RMSNorm Triton builds |
| [`rmsnorm`](https://huggingface.co/kernels-community/rmsnorm) | xpu,cpu | **P2** | Intel XPU RMSNorm (transformers default on XPU) |
| [`tinygrad-rms`](https://huggingface.co/kernels-community/tinygrad-rms) | cuda | **P2** | Alternative RMSNorm |
| [`relu`](https://huggingface.co/kernels-community/relu) | all | **P2** | Cross-platform ReLU (highest download count — generic) |

### SSM / linear attention / RWKV

| Hub repo | Drivers | AutoModel | Notes |
|---|---|---|---|
| [`mamba-ssm`](https://huggingface.co/kernels-community/mamba-ssm) | cuda | **P0** | `selective_scan_fn`, `causal_conv1d_*`, `Mamba2` — replaces `mamba-ssm` + `causal-conv1d` pip |
| [`causal-conv1d`](https://huggingface.co/kernels-community/causal-conv1d) | cuda | **P1** | Standalone conv1d (also bundled in `mamba-ssm`) |
| [`fla`](https://huggingface.co/kernels-community/fla) | cuda | **P1** | Flash-linear-attention ops — replaces `flash-linear-attention` pip for GDN/linear-attn models |
| [`rwkv`](https://huggingface.co/kernels-community/rwkv) | cuda | **P2** | RWKV time-mix kernels |

### MoE & GEMM

| Hub repo | Drivers | AutoModel | Notes |
|---|---|---|---|
| [`megablocks`](https://huggingface.co/kernels-community/megablocks) | cuda,rocm,xpu,cpu | **P1** | `MegaBlocksMoeMLP`, dropless MoE — transformers default MoE MLP; **not** DeepEP-compatible |
| [`megablocks-rocm`](https://huggingface.co/kernels-community/megablocks-rocm) | rocm | **P2** | ROCm megablocks variant |
| [`triton-kernels`](https://huggingface.co/kernels-community/triton-kernels) | cuda | **P1** | Triton MoE routing/SwiGLU (`swiglu`, `routing`) — mxfp4/bf16 |
| [`triton-moe`](https://huggingface.co/kernels-community/triton-moe) | — | **P2** | MoE-specific Triton ops |
| [`scattermoe`](https://huggingface.co/kernels-community/scattermoe) | cuda,rocm,xpu | **P2** | Scatter MoE kernels |
| [`sonic-moe`](https://huggingface.co/kernels-community/sonic-moe) | cuda | **P2** | `KernelBackendMoE`, routing kernels |
| [`vllm-moe`](https://huggingface.co/kernels-community/vllm-moe) | cuda | **P2** | vLLM MoE kernels |
| [`deep-gemm`](https://huggingface.co/kernels-community/deep-gemm) | cuda | **P2** | DeepSeek GEMM kernels |
| [`gemm`](https://huggingface.co/kernels-community/gemm) | — | **P2** | Generic GEMM |
| [`triton-scaled-mm`](https://huggingface.co/kernels-community/triton-scaled-mm) | — | **P2** | Scaled matmul (quant) |

### Quantization & FP8

| Hub repo | Drivers | AutoModel | Notes |
|---|---|---|---|
| [`quantization-bitsandbytes`](https://huggingface.co/kernels-community/quantization-bitsandbytes) | cpu | **P1** | BNB quant kernels — evaluate vs pip `bitsandbytes` for QLoRA |
| [`quantization-gptq`](https://huggingface.co/kernels-community/quantization-gptq) | cpu | **P2** | GPTQ CPU kernels |
| [`quantization-eetq`](https://huggingface.co/kernels-community/quantization-eetq) | cuda | **P2** | EETQ quant |
| [`finegrained-fp8`](https://huggingface.co/kernels-community/finegrained-fp8) | cuda,rocm,xpu | **P2** | FP8 quant kernels |
| [`fp8-fbgemm`](https://huggingface.co/kernels-community/fp8-fbgemm) | cuda,rocm,xpu | **P2** | FBGEMM FP8 |

### Vision / diffusion / domain-specific

| Hub repo | Drivers | AutoModel | Notes |
|---|---|---|---|
| [`deformable-detr`](https://huggingface.co/kernels-community/deformable-detr) | cuda | **P2** | Deformable attention — transformers default for DETR-style models |
| [`cv-utils`](https://huggingface.co/kernels-community/cv-utils) | cuda | **P2** | CV helper kernels |
| [`trimul-gpumode`](https://huggingface.co/kernels-community/trimul-gpumode) | cuda,rocm,xpu | **P2** | AlphaFold TriMul |
| [`punica-sgmv`](https://huggingface.co/kernels-community/punica-sgmv) | cuda | **P2** | Multi-LoRA SGMV (serving) |

### Platform-specific (Metal / ROCm AITER / GPT-OSS)

| Hub repo | Drivers | AutoModel | Notes |
|---|---|---|---|
| [`gpt-oss-triton-kernels`](https://huggingface.co/kernels-community/gpt-oss-triton-kernels) | cuda,rocm,xpu | **P1** | GPT-OSS model-specific Triton kernels |
| [`gpt-oss-metal-kernels`](https://huggingface.co/kernels-community/gpt-oss-metal-kernels) | metal | **P2** | GPT-OSS on Apple Metal |
| [`mlx-rmsnorm`](https://huggingface.co/kernels-community/mlx-rmsnorm) | metal | **P2** | MLX RMSNorm |
| [`mlx-quantization-metal-kernels`](https://huggingface.co/kernels-community/mlx-quantization-metal-kernels) | metal | **P2** | MLX quant |
| [`bitsandbytes-mps`](https://huggingface.co/kernels-community/bitsandbytes-mps) | metal | **P2** | BNB on MPS |
| [`aiter-kernels`](https://huggingface.co/kernels-community/aiter-kernels) | rocm | **P2** | AMD AITER kernel bundle |

### Optimizer / misc

| Hub repo | Drivers | AutoModel | Notes |
|---|---|---|---|
| [`adam-atan2`](https://huggingface.co/kernels-community/adam-atan2) | — | **P2** | Optimizer kernel |

### Suggested `kernels lock` set for AutoModel containers

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

Extended set for MoE + linear-attention + native-model experiments:

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

### Phase 0 — Validate passthrough (1–2 days)

**Goal:** Confirm Hub kernels work end-to-end through existing `NeMoAutoModel` without refactors.

- [ ] Install `kernels` in dev container
- [ ] Run a minimal finetune with `attn_implementation=kernels-community/flash-attn2` (no `flash-attn` pip)
- [ ] Run same with `attn_implementation=flash_attention_2` and no compiled FA — verify transformers Hub fallback
- [ ] Document working recipe YAML

### Phase 1 — Hub-aware availability & passthrough (small PR)

**Goal:** Make AutoModel's kernel layer Hub-aware without breaking existing installs.

1. **Add optional dependency group** (extend or alias `fa`):

   ```toml
   hub_kernels = ["kernels>=0.11.0"]
   fa = ["flash-attn<=2.8.3"]  # keep for TE-compat / ring-attn ABI pinning
   fa_or_hub = ["nemo_automodel[hub_kernels]"]  # Hub-only path
   ```

2. **Use `nemo_automodel/components/kernels/hub.py`** (added on this branch) — thin, cached wrapper around upstream `kernels.get_kernel` / `has_kernel`.

3. **Update `kernel_patches.py`:**
   - Extend `has_flash_attn()` to check Hub via `has_kernel()` / transformers `is_kernels_available()`
   - Teach `_apply_preload_overrides` that `kernels-community/flash-attn*` counts as flash for packed sequences
   - Extend `_get_next_fallback_attn` to handle Hub repo IDs (or delegate to transformers' resolver)

4. **Plumb `use_kernels` and `kernel_config`** through `NeMoAutoModel.from_pretrained` kwargs (pass-through, no new API surface beyond forwarding).

5. **Gate Liger:** when `use_kernels=True`, skip `_patch_liger_kernel` (transformers handles it).

### Phase 2 — Replace direct `flash_attn` imports (medium PR)

**Goal:** CP, speculative, and VLM paths load varlen kernels from Hub when pip package absent.

| File | Change |
|---|---|
| `blockdiag_cp/kernels.py` | Load `flash_attn_varlen_func` via `hub.py` |
| `kimivl/model.py`, `bagel/modeling_qwen2_packed.py` | Same |
| `ring_attention.py` | **High risk** — evaluate Hub FA2 private API compatibility; may keep pip pin for EAGLE |

### Phase 3 — Liger consolidation (medium PR)

- Default `use_kernels=True` for HF models when `kernels` installed
- Deprecate direct `liger_kernel` import path
- Add `kernels lock` to CI/container for reproducible builds

### Phase 4 — Native `BackendConfig` (optional, larger scope)

Extend `BackendConfig.attn` with Hub-backed options or a separate `hub_kernels: KernelConfig` field for NeMo-native model layers. This is a larger design decision — native models currently bypass transformers' attention dispatch entirely.

---

## Container / CI Implications

| Today | With Hub kernels |
|---|---|
| Docker builds compile `flash-attn` from source (slow, fragile) | `pip install kernels` + `kernels lock` / `kernels download` at build time |
| `fa` optional extra for users | `hub_kernels` extra; `fa` kept for TE version pinning |
| Per-CUDA wheel matrix in CI | Hub resolves compatible build variant at runtime |
| Air-gapped clusters | Pre-download with `kernels download` + lockfile in image |

Recommended container change:

```dockerfile
RUN uv pip install "kernels>=0.11.0" \
 && kernels lock kernels-community/flash-attn2 kernels-community/liger-kernels \
 && kernels download  # bake into image for offline use
```

---

## Risks & Blockers

| Risk | Severity | Mitigation |
|---|---|---|
| EAGLE ring attention pins FA 2.8.x private API | **High** | Keep `fa` dep for speculative; Hub path for standard HF attention only |
| TE requires specific `flash-attn` version | **High** | Document mutual exclusion; TE path keeps compiled FA |
| FSDP2 + Hub kernel autograd | **Medium** | Parity tests on 1-GPU and multi-GPU finetune |
| Hub download at training start (latency) | **Low** | `kernels lock` + pre-download in container |
| `USE_HUB_KERNELS=0` env disables all Hub loading | **Low** | Document; default YES in transformers |
| Packed sequence + CP override forces SDPA | **Medium** | Existing limitation; Hub FA varlen may enable CP+packing later |
| tilelang / FA4 / apache-tvm-ffi conflict | **High** | Unrelated to Hub; existing pyproject constraint remains |

---

## Recommended Next Steps

1. **Phase 0 spike** — run one L0 finetune on branch with `kernels` installed, no `flash-attn` pip, `attn_implementation=kernels-community/flash-attn2`.
2. **Phase 1 PR** — hub-aware availability checks + kwargs passthrough (minimal, backward compatible).
3. **Decision point** — after EAGLE ring-attn parity test, decide whether speculative stays on compiled FA permanently.
4. **Container experiment** — build image without `flash-attn` source compile, Hub-only FA2; measure build time and runtime parity.

---

## References

- [huggingface/kernels repo](https://github.com/huggingface/kernels) — loader source, CLI, lockfiles
- [Upstream integration guide](https://github.com/huggingface/kernels/blob/main/kernel-builder/skills/cuda-kernels/references/huggingface-kernels-integration.md)
- [Upstream example script](https://github.com/huggingface/kernels/blob/main/kernel-builder/skills/cuda-kernels/scripts/huggingface_kernels_example.py)
- [Kernels docs — Quickstart](https://huggingface.co/docs/kernels/en/basic-usage)
- [Transformers — Loading kernels](https://huggingface.co/docs/transformers/main/kernel_doc/loading_kernels)
- [Transformers `hub_kernels.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py)
- [kernels-community catalog (54 kernels)](https://huggingface.co/kernels-community/kernels)
- [kernels-community source repo](https://github.com/huggingface/kernels-community)
- AutoModel: `nemo_automodel/components/kernels/hub.py`, `nemo_automodel/_transformers/kernel_patches.py`, `nemo_automodel/components/models/common/utils.py` (`BackendConfig`)
