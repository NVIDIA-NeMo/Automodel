# MoE Configuration Guide for Automodel

This guide explains the Mixture-of-Experts (MoE) configuration settings in Automodel and how to use them effectively for training and fine-tuning MoE models.

## Overview

Automodel supports a wide range of MoE architectures including Qwen3-MoE, DeepSeek-V3/V3.2, GPT-OSS, Nemotron-V3, GLM4-MoE, Mixtral, Step-3.5 Flash, and MiniMax-M2. MoE configuration spans three areas:

1. **Distributed settings** (`distributed:`) — parallelism dimensions including expert parallelism
2. **Backend settings** (`model.backend:`) — kernel and dispatcher selection
3. **Model-intrinsic settings** — expert count, routing, activation functions (defined in the HuggingFace model config and mapped to `MoEConfig` at runtime)

---

## 1. Expert Parallelism (EP)

Expert parallelism shards experts across GPUs so that each GPU holds a subset of experts. It is controlled by the `ep_size` field in the `distributed:` section.

### Configuration

```yaml
distributed:
  strategy: fsdp2
  tp_size: 1
  cp_size: 1
  pp_size: 1
  ep_size: 8   # number of GPUs in the EP group
```

### How EP Works

With `ep_size: N`, each GPU holds `n_routed_experts / N` experts. Tokens are dispatched to the appropriate GPU via all-to-all communication (see [Token Dispatchers](#token-dispatchers) below).

**Constraint: `n_routed_experts` must be divisible by `ep_size`.** The framework enforces this at initialization:

```python
# From nemo_automodel/components/moe/parallelizer.py
assert model.model.moe_config.n_routed_experts % moe_mesh[ep_axis_name].size() == 0
```

### Interaction with Other Parallelism Dimensions

| Parallelism | Compatibility with EP | Notes |
|---|---|---|
| **FSDP2** | Fully supported | Expert parameters are sharded on dim=1 (token dimension) via EP shard mesh. Non-expert parameters use standard FSDP. |
| **PP** (pipeline) | Supported | Large models use PP+EP together. Example: Qwen3-235B uses `pp_size: 4, ep_size: 32`. |
| **CP** (context) | Supported with caveats | The MoE layer receives the CP mesh for aggregating auxiliary loss across CP ranks. **Requires `rope_fusion: false`** — see [Common Pitfalls](#6-common-pitfalls). |
| **TP** (tensor) | **Not supported** | Automodel asserts `tp_size == 1` for custom MoE models. |

### EP + FSDP Shard Mesh

When `ep_size < dp_size * cp_size`, expert parameters are additionally sharded across replicas using an EP shard mesh (FSDP on dim=1):

```
ep_shard_size = (dp_size * cp_size) / ep_size
```

The constraint `dp_size * cp_size` must be divisible by `ep_size` is enforced at mesh creation.

For very large models where expert params are already fully sharded by EP alone, set `reshard_after_forward: true` in the MoE config to free memory between forward/backward:

```yaml
distributed:
  ep_size: 64
  moe:
    reshard_after_forward: true
```

### Real-World Examples

| Model | Nodes | EP | PP | Config |
|---|---|---|---|---|
| Qwen3-MoE 30B (128 experts) | 1 | 8 | 1 | `examples/benchmark/configs/qwen3_moe_30b_te_deepep.yaml` |
| Qwen3-MoE 235B (128 experts) | 32 | 32 | 4 | `examples/benchmark/configs/qwen3_moe_235b_te_deepep.yaml` |
| DeepSeek-V3 (256 experts) | 32 | 64 | 4 | `examples/benchmark/configs/deepseek_v3_te_deepep.yaml` |
| GPT-OSS 120B | 8 | 64 | 1 | `examples/benchmark/configs/gptoss_120b_te_deepep.yaml` |
| Step-3.5 Flash | 8 | 32 | 2 | `examples/llm_finetune/stepfun/step_3.5_flash_hellaswag_pp.yaml` |

---

## 2. Expert Routing

Expert routing determines how tokens are assigned to experts. These parameters are defined in the model's HuggingFace config and mapped to `MoEConfig` fields at model construction time.

### Core Routing Parameters

| MoEConfig Field | Description | Typical Values |
|---|---|---|
| `n_routed_experts` | Total number of routed experts | 64–256 |
| `n_shared_experts` | Number of shared (always-active) experts | 0–2 |
| `n_activated_experts` | Experts activated per token (top-k) | 2–8 |
| `score_func` | Routing score function | `"softmax"`, `"sigmoid"` |
| `norm_topk_prob` | Normalize routing probabilities after top-k | `true`/`false` |
| `route_scale` | Scaling factor applied to routing scores | 1.0–2.5 |
| `aux_loss_coeff` | Load balancing auxiliary loss coefficient | 0.0–0.001 |
| `softmax_before_topk` | Apply softmax before top-k selection | `true`/`false` |

### Routing Score Functions

**Softmax routing** (Qwen3-MoE, GPT-OSS, Mixtral): Computes softmax over all expert logits, then selects top-k. Standard approach with well-understood training dynamics.

**Sigmoid routing** (DeepSeek-V3, Nemotron-V3, GLM4-MoE): Computes independent sigmoid scores per expert, then selects top-k. Often combined with grouped routing and `route_scale > 1`.

### Grouped Routing

Some models partition experts into groups for hierarchical routing:

| MoEConfig Field | Description |
|---|---|
| `n_expert_groups` | Number of expert groups (0 = no grouping) |
| `n_limited_groups` | Select experts only from top-k groups |

DeepSeek-V3 example: 256 experts in 8 groups, selecting from top 4 groups (`n_expert_groups: 8`, `n_limited_groups: 4`).

### Load Balancing

**Auxiliary loss** penalizes uneven expert utilization. Controlled by `aux_loss_coeff`:

- `0.0` — no load balancing loss (rely on other mechanisms)
- `0.001` — typical value for DeepSeek-V3-style models

**Expert bias correction** (DeepSeek-V3 style): A bias term is maintained per expert and updated based on cumulative load:

```
new_bias = old_bias + gate_bias_update_factor * sign(avg_load - expert_load)
```

Relevant fields: `gate_bias_update_factor`, `router_bias`, `force_e_score_correction_bias`.

### Shared Experts

Shared experts process all tokens (not routed). They run in parallel with routed experts:

| MoEConfig Field | Description |
|---|---|
| `n_shared_experts` | Number of shared experts (0 = none) |
| `shared_expert_gate` | Whether shared experts have a learned gating mechanism |
| `shared_expert_inter_dim` | Intermediate dimension for shared experts |
| `shared_expert_activation` | Activation function for shared experts (`"swiglu"` or `"relu2"`) |

---

## 3. Expert Architecture

### Gated vs Non-Gated Experts

Expert FFN layers support two structural patterns based on the `expert_activation` field:

**Gated activations** (require 2x intermediate dimension):

- **SwiGLU** (`expert_activation: "swiglu"`): `silu(gate_proj(x)) * up_proj(x)` → `down_proj(...)`. Default for Qwen3-MoE, DeepSeek-V3.
- **Quick-GeGLU** (`expert_activation: "quick_geglu"`): `sigmoid(alpha * gate_proj(x)) * up_proj(x)` with clamping. Used by GPT-OSS. Additional parameters: `activation_alpha` (default 1.702), `activation_limit` (default 7.0).

Gated experts use separate `gate_proj` and `up_proj` projections (or a fused `gate_and_up_proj`). The HuggingFace config typically stores these as either:
- `gate_proj` + `up_proj` (separate weights)
- `gate_and_up_projs` (fused single weight, split at runtime)

**Non-gated activation** (require 1x intermediate dimension):

- **ReLU²** (`expert_activation: "relu2"`): `relu(up_proj(x))²` → `down_proj(...)`. Used by Nemotron-V3. 50% memory reduction compared to gated activations since there is no gate projection.

### Activation Function Summary

| Activation | `expert_activation` | Projections | Models |
|---|---|---|---|
| SwiGLU | `"swiglu"` | gate + up + down | Qwen3-MoE, DeepSeek-V3, GLM4-MoE |
| Quick-GeGLU | `"quick_geglu"` | gate + up + down | GPT-OSS |
| ReLU² | `"relu2"` | up + down | Nemotron-V3 |

### Intermediate Sizes

The `moe_inter_dim` field controls the expert FFN hidden dimension (typically different from the dense MLP `inter_dim`). For example, Qwen3-30B-A3B uses `moe_intermediate_size: 2560` for experts vs `intermediate_size: 18944` for dense layers.

### Latent Projections

Some models project inputs to a lower dimension before routing to experts. Controlled by `moe_latent_size`:

```python
# When moe_latent_size is set, expert input is projected down:
# input (dim) → fc1_latent_proj (moe_latent_size) → experts → fc2_latent_proj (dim)
```

---

## 4. Normalization — RMS Norm Options

The `rms_norm` backend setting in `model.backend` controls which RMSNorm implementation is used throughout the model, including within MoE layers.

### Available Backends

| Value | Implementation | Description |
|---|---|---|
| `"torch"` | Standard PyTorch RMSNorm | BF16 computation. Fastest but may have numerical issues with some models. |
| `"torch_fp32"` | `Float32RMSNorm` (torch.compiled) | Weights stay BF16 for FSDP2 compatibility; inputs are upcast to FP32 for computation. Best for training stability. |
| `"te"` | Transformer Engine fused kernel | Uses TE's optimized fused RMSNorm. Requires TE installation. |

### When to Use Each

- **`torch_fp32`**: Recommended for fine-tuning when training stability is critical. Required for Qwen3-MoE fine-tuning to avoid loss divergence. Also used for Step-3.5 Flash and DeepSeek-V3.2.
- **`te`**: Recommended for benchmarking and maximum throughput. Used in all benchmark configs (Qwen3-MoE, DeepSeek-V3, GPT-OSS, Nemotron-V3).
- **`torch`**: Fallback when neither TE nor FP32 precision is needed. Used in pure-torch configs.

### Example: Qwen3-MoE Fine-tuning (Stability)

```yaml
# From examples/llm_finetune/qwen/qwen3_moe_30b_te_packed_sequence_flashoptim.yaml
model:
  backend:
    rms_norm: torch_fp32   # FP32 RMSNorm for training stability
```

### Example: Qwen3-MoE Benchmarking (Throughput)

```yaml
# From examples/benchmark/configs/qwen3_moe_30b_te_deepep.yaml
model:
  backend:
    rms_norm: te           # TE fused kernel for speed
```

---

## 5. Kernel Backends

### Complete `BackendConfig` Reference

All fields in the `BackendConfig` dataclass (defined in `nemo_automodel/components/models/common/utils.py`):

| Field | Type | Default | Description |
|---|---|---|---|
| `attn` | `"te"`, `"sdpa"`, `"flex"` | `"te"` (GPU), `"sdpa"` (CPU) | Attention backend. `"te"` uses Transformer Engine, `"sdpa"` uses PyTorch scaled dot-product attention, `"flex"` uses FlexAttention. |
| `linear` | `"torch"`, `"te"` | `"te"` (GPU), `"torch"` (CPU) | Linear layer backend. `"te"` uses Transformer Engine fused linear. |
| `rms_norm` | `"torch"`, `"torch_fp32"`, `"te"` | `"torch_fp32"` | RMSNorm backend. See [Normalization](#4-normalization--rms-norm-options). |
| `rope_fusion` | `bool` | `true` (GPU) | Whether to use fused RoPE (requires TE). Must be `false` when `cp_size > 1`. |
| `experts` | `"torch"`, `"te"`, `"gmm"`, `"torch_mm"` | `"torch_mm"` (GPU), `"torch"` (CPU) | MoE expert GEMM backend (see below). |
| `dispatcher` | `"torch"`, `"deepep"` | `"deepep"` (if available), `"torch"` | MoE token dispatcher (see below). |
| `fake_balanced_gate` | `bool` | `false` | Replace learned Gate with synthetic balanced routing. **Benchmarking only.** |
| `fake_gate_noise` | `float` | `0.0` | Noise level [0, 1] for `FakeBalancedGate`. Higher values create more realistic imbalance. Approximate max/mean load ratios (64 experts, top-8, 4096 tokens): 0.0→1.00x, 0.1→~1.2x, 0.3→~1.6x, 0.5→~2.0x, 1.0→~2.8x. |
| `enable_hf_state_dict_adapter` | `bool` | `true` | Enable HuggingFace state dict adapter for checkpoint loading. |
| `enable_fsdp_optimizations` | `bool` | `false` | Enable FSDP2-specific optimizations. |
| `te_fp8` | `TEFp8Config \| None` | `None` | FP8 quantization config (see [FP8 Training](#fp8-training)). Requires `linear: te` or `experts: te`. |
| `gate_precision` | `str \| torch.dtype \| None` | `None` | Optional dtype override for gate computation (e.g., `"float32"` for higher-precision routing). |
| `enable_deepep` | `bool \| None` | `None` | **Deprecated.** Use `dispatcher: "deepep"` and `experts: "gmm"` instead. |

> **Validation rules** enforced by `BackendConfig.__post_init__`:
> - If `experts` is `"te"` or `"gmm"` but `dispatcher` is not `"deepep"`, the config auto-corrects to `dispatcher: "torch"` and `experts: "torch_mm"` with a warning.
> - `te_fp8` requires at least one TE backend (`linear: "te"` or `experts: "te"`).

### Expert GEMM Backends

The `experts` field selects the matrix multiplication implementation for expert computation:

| Value | Implementation | Description |
|---|---|---|
| `"torch"` | Per-expert loop | Simple loop over experts with gather/scatter. No external dependencies. Slowest. |
| `"torch_mm"` | `torch._grouped_mm` | PyTorch native grouped GEMM. Good default with no extra dependencies. |
| `"gmm"` | `grouped_gemm.ops.gmm` | Third-party grouped GEMM library. Recommended with DeepEP dispatcher. |
| `"te"` | TE `GroupedLinear` | Transformer Engine grouped linear with FP8 support. Highest throughput with FP8. |

### Token Dispatchers

The `dispatcher` field controls how tokens are communicated between EP ranks:

| Value | Implementation | Description |
|---|---|---|
| `"torch"` | DTensor all-gather/reduce-scatter | Standard PyTorch distributed. No extra dependencies. |
| `"deepep"` | DeepEP fused all-to-all | Optimized fused dispatch from DeepSeek. Best for cross-node MoE. Requires `deep_ep` package. |

### Recommended Pairings

| Scenario | `experts` | `dispatcher` | Notes |
|---|---|---|---|
| Maximum throughput | `gmm` | `deepep` | Most benchmark configs use this |
| Fine-tuning (no extra deps) | `torch_mm` | `deepep` | Good balance of speed and simplicity |
| Pure PyTorch (CPU/debug) | `torch` | `torch` | No CUDA or external dependencies needed |
| FP8 training | `te` | `deepep` | Enables FP8 quantized expert computation |

### Optimizer Choice: Adam vs FlashAdamW

The optimizer impacts MoE training memory footprint:

**Standard Adam** (`torch.optim.Adam`): Maintains FP32 master weights and two FP32 moment buffers per parameter. For large MoE models, optimizer state can dominate memory usage.

```yaml
# Standard Adam
optimizer:
  _target_: torch.optim.Adam
  lr: 1.0e-4
  betas: [0.9, 0.999]
```

**FlashAdamW** (`flashoptim.FlashAdamW`): Uses compressed master weights (configurable 8-bit/16-bit/24-bit correction terms) to reduce optimizer memory while maintaining training quality.

```yaml
# From examples/llm_finetune/qwen/qwen3_moe_30b_te_packed_sequence_flashoptim.yaml
optimizer:
  _target_: flashoptim.FlashAdamW
  lr: 1.0e-4
  betas: [0.9, 0.999]
  eps: 1e-7
  weight_decay: 0
  master_weight_bits: 24   # compressed master weights
```

FlashAdamW is particularly valuable for MoE models because the large number of expert parameters multiplies optimizer state memory.

### FP8 Training

Enable FP8 quantization for expert computation via `te_fp8`:

```yaml
model:
  backend:
    experts: te          # TE backend required for FP8
    te_fp8:
      recipe: current    # or "block" for block-level scaling
```

---

## 6. Common Pitfalls

### EP must divide `n_routed_experts`

`ep_size` must evenly divide the model's `n_routed_experts`. The framework will raise an assertion error otherwise.

```
# Bad: 128 experts with ep_size=6
AssertionError: n_routed_experts 128 must be divisible by expert_parallel_degree 6

# Good: 128 experts with ep_size=8 (128/8 = 16 experts per GPU)
```

### CP + EP requires `rope_fusion: false`

When using context parallelism (`cp_size > 1`) with EP, fused RoPE must be disabled. Automodel automatically disables it with a warning:

```python
# From nemo_automodel/recipes/llm/train_ft.py
if self.dist_setup.cp_size > 1 and self.cfg.get("model.backend.rope_fusion", False):
    logging.info("Disabling rope_fusion because cp_size=%d > 1", self.dist_setup.cp_size)
    self.cfg.model.backend.rope_fusion = False
```

Best practice: explicitly set `rope_fusion: false` in your config when using CP to avoid confusion.

### `dp_size * cp_size` must be divisible by `ep_size`

The EP mesh is carved out of the data-parallel and context-parallel dimensions. This constraint is enforced at mesh creation:

```python
dp_cp_size = dp_size * cp_size
assert dp_cp_size % ep_size == 0
```

### TP is not supported with MoE

Tensor parallelism (`tp_size > 1`) is not supported for custom MoE models and will raise an assertion error. Use EP instead for distributing expert computation.

### `reshard_after_forward` and `wrap_outer_model` for large models

For large MoE models using PP + EP, set these in the `distributed.moe` section to manage memory:

```yaml
distributed:
  pp_size: 4
  ep_size: 32
  moe:
    reshard_after_forward: false  # true if experts are memory-bound
    wrap_outer_model: false       # false when using PP to avoid double wrapping
```

### Choosing `rms_norm` for training stability

Using `rms_norm: te` (or `torch`) during fine-tuning can cause loss divergence for some models (notably Qwen3-MoE). Switch to `rms_norm: torch_fp32` for stable training:

```yaml
model:
  backend:
    rms_norm: torch_fp32   # required for Qwen3-MoE fine-tuning stability
```

### `fake_balanced_gate` for benchmarking only

Setting `fake_balanced_gate: true` replaces the learned router with a synthetic balanced gate. This is useful for throughput benchmarking but produces meaningless model outputs. Never use it for actual training:

```yaml
model:
  backend:
    fake_balanced_gate: true    # benchmarking only!
    fake_gate_noise: 0.0        # 0.0 = perfectly balanced, higher = more realistic imbalance
```

### FP8 requires at least one TE backend

The `te_fp8` setting requires at least one backend component to use TE (e.g., `linear: te` or `experts: te`).

---

## MoE Parallelizer Settings

The MoE parallelizer (`nemo_automodel/components/moe/parallelizer.py`) orchestrates how MoE models are distributed across GPUs. It is configured via the `distributed:` section and applies expert parallelism, FSDP, context parallelism, and activation checkpointing.

### Configuration

```yaml
distributed:
  strategy: fsdp2
  ep_size: 8
  cp_size: 1
  pp_size: 1
  activation_checkpointing: true
  moe:
    reshard_after_forward: false    # reshard expert weights after forward pass
    wrap_outer_model: true          # wrap outer model for FSDP (false for PP)
    ignore_router_for_ac: false     # save router outputs during activation checkpointing
```

### `parallelize_model()` — Main Entry Point

The `parallelize_model()` function applies all parallelism strategies to the model in order.

**Full signature:**

```python
def parallelize_model(
    model: torch.nn.Module,
    world_mesh: DeviceMesh,
    moe_mesh: DeviceMesh | None,
    *,
    dp_axis_names: tuple[str, ...],
    cp_axis_name: str | None = None,
    tp_axis_name: str | None = None,
    ep_axis_name: str | None = None,
    ep_shard_axis_names: tuple[str, ...] | None = None,
    activation_checkpointing: bool = False,
    ignore_router_for_ac: bool = False,
    reshard_after_forward: bool = False,
    lm_head_precision: str | torch.dtype | None = None,
    wrap_outer_model: bool = True,
    mp_policy: MixedPrecisionPolicy | None = None,
):
```

**Key user-facing parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `activation_checkpointing` | `bool` | `false` | Enable activation checkpointing for MoE layers. Reduces memory at the cost of recomputation during backward pass. |
| `ignore_router_for_ac` | `bool` | `false` | When `true`, uses selective checkpointing that preserves router outputs, avoiding recomputation of expert routing during backward. Beneficial when routing is expensive relative to expert computation. |
| `reshard_after_forward` | `bool` | `false` | Reshard expert parameters after the forward pass to free memory. Useful for very large models where expert params are memory-bound. |
| `wrap_outer_model` | `bool` | `true` | Wrap the outer model with FSDP. Set to `false` when using pipeline parallelism (`pp_size > 1`) to avoid double wrapping. |
| `mp_policy` | `MixedPrecisionPolicy` | `None` | Mixed precision policy for FSDP. When `None`, defaults to BF16 params/output with FP32 reduce and `cast_forward_inputs=True`. |
| `lm_head_precision` | `str \| torch.dtype` | `None` | Optional custom precision for the language model head. When set to `"float32"`, uses FP32 for param, reduce, and output dtypes on the LM head. |

**Mesh axis parameters** (configured internally based on `distributed:` settings):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dp_axis_names` | `tuple[str, ...]` | — (required) | Data parallelism axis names in the device mesh. |
| `cp_axis_name` | `str \| None` | `None` | Context parallelism axis name. Set when `cp_size > 1`. |
| `tp_axis_name` | `str \| None` | `None` | Tensor parallelism axis name. Currently unsupported for MoE (`tp_size` must be 1). |
| `ep_axis_name` | `str \| None` | `None` | Expert parallelism axis name. Set when `ep_size > 1`. |
| `ep_shard_axis_names` | `tuple[str, ...] \| None` | `None` | Axes for additional FSDP sharding of expert parameters when `ep_size < dp_size * cp_size`. |

### `apply_ep()` — Expert Parallelism

Distributes experts across GPUs using `DTensor` with the `ExpertParallel` placement strategy. Shards grouped expert weight tensors on dimension 0 (the expert dimension) and configures the token dispatcher with the MoE mesh.

Expert types are handled differently:
- **`GroupedExpertsTE`**: Initializes the token dispatcher without DTensor wrapping (TE manages its own distribution).
- **Other expert types**: Uses `distribute_module` with the `ExpertParallel` style, which also handles `GroupedExpertsDeepEP` by calling `init_token_dispatcher(ep_mesh=device_mesh)`.

**Constraint:** `n_routed_experts` must be divisible by `ep_size`.

### `apply_fsdp()` — FSDP for MoE

Applies Fully Sharded Data Parallel to the model with MoE-aware wrapping:

- **Expert parameters** are sharded on the EP shard mesh (dim=1, token dimension) when `ep_shard_enabled` is true (i.e., `ep_size < dp_size * cp_size`).
- **Non-expert parameters** use standard FSDP on the main data-parallel mesh.

**Additional parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `offload_policy` | `OffloadPolicy \| None` | `None` | Parameter offload policy for CPU offloading to reduce GPU memory. |

**Default mixed precision policy** (when `mp_policy` is `None`):

```python
MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    output_dtype=torch.bfloat16,
    cast_forward_inputs=True,
)
```

**Components wrapped by FSDP:**
- Transformer blocks (with expert parameters excluded if EP is enabled)
- `embed_tokens` (if present)
- `lm_head` (with optional custom precision via `lm_head_precision`)
- `audio_tower` (if present and has trainable parameters — for audio-language models)
- `visual` (if present and has trainable parameters — for vision-language models)

### `apply_ac()` — Activation Checkpointing

When `ignore_router_for_ac: false` (default), wraps MoE block layers with standard `torch.utils.checkpoint`. When `true`, uses a selective policy that saves router outputs (based on tensor shape matching `[*, hidden_size, num_experts]`), avoiding recomputation of the routing decision.

The `hidden_size` and `num_experts` values for selective checkpointing are auto-derived from the model config:
- `hidden_size`: tries `model.config.text_config.hidden_size` (VLM models) then `model.config.hidden_size`
- `num_experts`: tries `model.model.moe_config.n_routed_experts`, then config attributes `num_experts`, `moe_num_experts`, `n_routed_experts` in order

### `apply_cp()` — Context Parallelism

Applies context parallelism to attention layers. The `cp_comm_type` parameter (default `"p2p"`) controls the communication pattern used by TE attention modules for CP.

**Layer handling:**
- **Full attention layers**: Sets context parallel group on TE's `DotProductAttention`
- **Linear attention layers**: Stores `cp_mesh` on the module for custom CP logic
- **MoE modules**: Sets `moe_module.cp_mesh` for aggregating auxiliary loss across CP ranks
- Sets `model._cp_enabled = True` to enable attention mask nulling in the forward pass

---

## Load Balance Metrics & Visualization

Monitoring expert load balance is critical for MoE training — unbalanced routing leads to wasted capacity (underloaded experts), bottlenecks (overloaded experts), and in extreme cases "dead experts" that receive zero tokens. Automodel provides built-in load balance tracking, metrics computation, and W&B integration.

> **See also:** [GitHub Discussion #1266](https://github.com/NVIDIA-NeMo/Automodel/discussions/1266) for community examples and visualizations.

### Configuration

Enable load balance metrics via the `moe_metrics` section in your training config:

```yaml
moe_metrics:
  enabled: true               # enable load balance tracking
  mode: brief                 # "brief" = aggregated scalars, "detailed" = per-layer breakdowns
  detailed_every_steps: 100   # log detailed metrics every N steps (only when mode="detailed")
  top_k_experts: 5            # emit top/bottom N experts by utilization (0 = disable per-expert logging)
```

**`MoEMetricsConfig` fields** (from `nemo_automodel/components/moe/config.py`):

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` | Enable load balance metric tracking. Adds negligible overhead (one `.detach()` copy per Gate per forward pass). |
| `mode` | `str` | `"brief"` | `"brief"` emits aggregated scalars only; `"detailed"` adds per-layer breakdowns. |
| `detailed_every_steps` | `int \| None` | `None` | When `mode="detailed"`, emit detailed metrics every N steps. Falls back to brief metrics on other steps. `None` = every step. |
| `top_k_experts` | `int` | `0` | Number of top (highest) and bottom (lowest) utilization experts to log. Keeps W&B key count bounded to `2 * top_k` regardless of model size. `0` = disable per-expert logging. |

### Available Metrics

#### Brief Mode

Brief mode produces aggregated scalar metrics suitable for monitoring training runs at a glance:

| Metric Key | Description |
|---|---|
| `moe/cv_mean`, `moe/cv_median`, `moe/cv_min`, `moe/cv_max` | Coefficient of variation (std/mean) of expert loads, aggregated across all MoE layers. Lower is better — 0.0 means perfectly balanced. |
| `moe/aux_loss_mean` | Auxiliary load balancing loss averaged across layers (only when `aux_loss_coeff > 0`). |
| `moe/expert_utilization_p25`, `_median`, `_p75`, `_min`, `_max` | Utilization ratio percentiles across all experts globally. 1.0 = ideal load; >1 = overloaded; <1 = underloaded; 0 = dead. |
| `moe/dead_expert_frac_mean` | Fraction of experts receiving zero tokens, averaged across layers. |
| `moe/expert_diversity_mean`, `moe/expert_diversity_min` | Shannon entropy-based diversity: `exp(H) / N` where H is entropy of routing distribution. 1.0 = all experts equally used; low values = collapsed routing. |
| `moe_expert_utilization/layer_{i}_expert_{j}` | Individual utilization ratios for the top-K highest and bottom-K lowest experts globally. |

#### Detailed Mode (adds per-layer breakdowns)

When `mode: "detailed"`, all brief metrics are emitted plus per-layer keys:

| Metric Key | Description |
|---|---|
| `moe/layer_{i}/cv` | Coefficient of variation for layer i. |
| `moe/layer_{i}/aux_loss` | Auxiliary loss for layer i. |
| `moe/layer_{i}/utilization_mean` | Mean utilization ratio for layer i. |
| `moe/layer_{i}/dead_expert_frac` | Dead expert fraction for layer i. |
| `moe/layer_{i}/expert_diversity` | Expert diversity score for layer i. |

### Auxiliary Loss Configuration

The auxiliary loss is the primary training-time mechanism for encouraging balanced routing. It is configured via `MoEConfig.aux_loss_coeff` (set in the model's HuggingFace config):

```
aux_loss = aux_loss_coeff * sum(f_i * P_i)
```

Where:
- `f_i = expert_load[i] * n_experts / (top_k * context_length)` — normalized load fraction
- `P_i = expert_scores[i] / context_length` — average routing probability per expert

| `aux_loss_coeff` | Effect |
|---|---|
| `0.0` | No auxiliary loss — rely on expert bias correction or other mechanisms. |
| `0.001` | Typical for DeepSeek-V3/V3.2-style models. Gentle load balancing. |
| `0.01` | Stronger load balancing for models with severe imbalance. |

The loss is automatically scaled during backpropagation via `MoEAuxLossAutoScaler` to maintain stable gradients relative to the main loss.

### Visualization with Weights & Biases

Metrics are logged directly to W&B via the training recipe. The integration works as follows:

1. **Collection phase** (`_collect_moe_load_balance`): After each training step, expert loads are all-reduced across the DP group so metrics reflect global routing, not a single rank's view.
2. **Logging phase** (`_log_moe_metrics`): On the logging rank, metrics are computed and passed to `wandb.log()`.

**Recommended W&B panel setup:**

| Panel | Metric Keys | Purpose |
|---|---|---|
| Load Balance Overview | `moe/cv_mean`, `moe/cv_median` | Track overall routing balance over training |
| Expert Health | `moe/dead_expert_frac_mean`, `moe/expert_diversity_mean` | Detect expert collapse early |
| Aux Loss | `moe/aux_loss_mean` | Monitor load balancing loss convergence |
| Utilization Distribution | `moe/expert_utilization_p25`, `_median`, `_p75` | Understand the spread of expert utilization |
| Hot/Cold Experts | `moe_expert_utilization/*` | Identify specific overloaded/underloaded experts |

### Interpretation Guidelines

| Metric | Healthy Range | Warning Signs |
|---|---|---|
| `cv_mean` | < 0.3 | > 0.5 indicates significant imbalance; > 1.0 indicates severe skew |
| `dead_expert_frac_mean` | 0.0 | > 0.05 (5%) means some experts are wasted; investigate routing |
| `expert_diversity_mean` | > 0.7 | < 0.5 indicates routing collapse — many experts are underutilized |
| `expert_utilization_max` | < 2.0 | > 3.0 indicates extreme hotspotting on certain experts |
| `expert_utilization_min` | > 0.3 | < 0.1 (or 0.0) indicates near-dead experts |
| `aux_loss_mean` | Decreasing over training | Flat or increasing suggests `aux_loss_coeff` is too low or routing is stuck |

### Common Tuning Tips

1. **Start with brief mode**: Use `mode: "brief"` during initial training runs. Switch to `"detailed"` only when debugging specific layer-level issues.

2. **Set `top_k_experts` thoughtfully**: For models with many experts (128–256), set `top_k_experts: 5–10` to keep W&B dashboards manageable. The top/bottom experts are the ones that matter for diagnosis.

3. **Use `detailed_every_steps` to reduce logging overhead**: For long training runs with detailed mode, `detailed_every_steps: 100` logs per-layer metrics every 100 steps while emitting brief metrics on intervening steps.

4. **Tune `aux_loss_coeff` based on `cv_mean`**: If `cv_mean` remains high (> 0.5) after initial training, increase `aux_loss_coeff`. If training loss is unstable, decrease it. The auxiliary loss should be roughly 1–2 orders of magnitude smaller than the main loss.

5. **Watch for dead experts early**: Check `dead_expert_frac_mean` in the first few hundred steps. Dead experts that appear early rarely recover. Consider increasing `aux_loss_coeff` or using expert bias correction (`gate_bias_update_factor > 0`).

6. **Expert bias correction as an alternative**: For models that use sigmoid routing (DeepSeek-V3, Nemotron-V3), expert bias correction via `gate_bias_update_factor` can be more effective than auxiliary loss alone. The two mechanisms can be used together.

---

## Quick Reference: Complete MoE Config Example

```yaml
# Full example: Qwen3-MoE 30B fine-tuning on 8 GPUs
distributed:
  strategy: fsdp2
  tp_size: 1          # TP not supported with MoE
  cp_size: 1
  pp_size: 1
  ep_size: 8          # 128 experts / 8 GPUs = 16 experts per GPU
  activation_checkpointing: true

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen3-30B-A3B
  backend:
    attn: te
    linear: te
    rms_norm: torch_fp32     # FP32 for training stability
    rope_fusion: true        # OK when cp_size=1
    experts: torch_mm        # PyTorch grouped GEMM
    dispatcher: deepep       # DeepEP for efficient dispatch
    fake_balanced_gate: false  # real routing for training
    enable_hf_state_dict_adapter: true

optimizer:
  _target_: flashoptim.FlashAdamW
  lr: 1.0e-4
  betas: [0.9, 0.999]
  master_weight_bits: 24
```

## Further Reading

- [Large MoE Fine-tuning Guide](guides/llm/large_moe_finetune.md) — step-by-step recipes for GLM-5, MiniMax-M2.5, Step-3.5 Flash, DeepSeek-V3.2
- [Megatron MoE README](../nemo_automodel/components/moe/megatron/README.md) — technical details on token dispatching and load balancing
- Example configs in `examples/benchmark/configs/` and `examples/llm_finetune/`
