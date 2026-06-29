# Titans memory notes (linear + deep)

First-class **Titans** memory model for NeMo AutoModel. One `NeuralMemory` token
mixer supports two memory kinds behind a shared API:

- **`mem_depth=1` (linear memory):** **Gated DeltaNet (GDN) + data-dependent
  momentum + (forget) decay gate**, wrapping
  [`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) (fla)
  kernels.
- **`mem_depth>=2` (deep memory):** a per-head MLP whose weights are updated by
  **chunkwise test-time gradient descent** with the same momentum + forget gates
  (see the *Deep memory* section below).

## What this is

The token mixer is `NeuralMemory` (`layers.py`): a per-head matrix memory
`S ∈ R^{mem_dim×mem_dim}` updated by the gated delta rule **plus a
data-dependent momentum over the per-token surprise**. Per head, per step `t`:

```
alpha_t   = exp(g_t)                                 # forget / decay gate, g_t <= 0
pred_t    = (alpha_t * S_{t-1})^T k_t                # readout of the (decayed) memory for k_t
surprise_t = k_t ⊗ [ beta_t (v_t - pred_t) ]         # delta-rule surprise (outer product)
M_t       = eta_t * M_{t-1} + surprise_t             # data-dependent momentum carry
S_t       = alpha_t * S_{t-1} + M_t                  # write to memory
o_t       = S_t^T q_t                                # retrieve
```

- `alpha_t = exp(g_t)` is the GDN decay gate, with `g_t = -exp(A_log)·softplus(a_t + dt_bias)`
  (same parameterization as Qwen3-Next GDN; `a_t` is a data-dependent projection).
- `beta_t = sigmoid(b_t)` is the delta-rule surprise step size (learning rate).
- `eta_t = sigmoid(m_t) ∈ (0,1)` is the **data-dependent momentum** (the Titans addition).

**Reduction to Gated DeltaNet.** Setting `eta_t = 0` makes `M_t = surprise_t`, so
`S_t = alpha_t·S_{t-1} + surprise_t`, which is *exactly* the gated delta rule. This
is verified numerically against fla's `chunk_gated_delta_rule` (see Validation),
proving Titans generalizes GDN (`momentum=False` ⇒ GDN). The `forget=False` toggle
additionally fixes `alpha_t = 1` (pure, non-gated delta rule).

## What wraps fla

- **`momentum=False` (GDN path):** delegates directly to
  `fla.ops.gated_delta_rule.chunk_gated_delta_rule` (fused Triton chunk kernel).
  This is the fast, production GDN path.
- **`momentum=True` (Titans path):** `titans_delta_rule_recurrence`, a pure-torch
  O(T) sequential recurrence that mirrors fla's `naive_recurrent_gated_delta_rule`
  convention exactly and adds the momentum carry `M`. It is autograd-friendly
  (trains fine) but not yet a fused kernel.
  - **Known limitation / Phase 2 hook:** the momentum path is the naive recurrence,
    not a chunked Triton kernel. A chunked momentum kernel (and the deep-memory
    kernel) is the Phase-2 work; the `chunk_size` config field is threaded through
    for it. `q,k` are L2-normalized before the recurrence to match fla's
    `use_qk_l2norm_in_kernel=True`; the query scale is `1/sqrt(mem_dim)` (fla default).

fla is an optional dependency (`flash-linear-attention>=0.4.2`, the `[fla]` extra),
imported via `safe_import`. If fla is unavailable, the GDN path falls back to the
naive recurrence with `eta=0`.

## Precision (fp32 decay-gate params)

`A_log` and `dt_bias` are intrinsically fp32 — `A_log` is exponentiated, so bf16
rounding becomes a proportional error on the decay rate that the recurrence
compounds across the sequence. They are stored fp32 and the gate is computed in
fp32 regardless of compute dtype. `TitansForCausalLM` declares
`_keep_in_fp32_modules{,_strict} = ["A_log", "dt_bias"]`, and the state-dict
adapter upcasts them on load and forces `F32` on HF export. (The full FSDP2
mixed-precision `_fp32_params` *holder* routing used by Qwen3-Next is the eventual
production form; Phase 1 keeps the params fp32 directly, which is sufficient for
FSDP2 with `_keep_in_fp32_modules`.)

## Deep memory (`mem_depth>=2`) — chunkwise test-time gradient descent

The same `NeuralMemory` module also implements the **deep (MLP) memory** of
Titans for `mem_depth>=2`. The memory is now a per-head MLP whose *weights are the
state*, updated online by **test-time gradient descent** on the associative loss
`(1/m)‖M_W(k_t) - v_t‖²`. The Phase-1 projections (`q/k/v_proj`) and per-head
scalar gates are reused unchanged; the gate→update mapping is:

| deep update term | Phase-1 gate | note |
|---|---|---|
| learning rate `θ_t` (`s_t = -θ_t·∇_W`) | `beta = sigmoid(b_proj)` | delta-rule step |
| momentum `η_t` (`S_t = η_t·S_{t-1} + s_t`) | `eta = sigmoid(m_proj)` | data-dependent momentum |
| forget keep-factor (`W_t = keep_t·W_{t-1} + S_t`) | `exp(g)` | GDN decay (`g = -exp(A_log)·softplus(a+dt_bias)`) |

Implementation (ported from the validated standalone kernel in
`deep-memory/deep_neural_memory.py`):

- **Per head via batch-folding.** q,k,v `[B,S,H,D]` and gates `[B,S,H]` fold the
  head axis into the batch (`B*H`) so the MLP recurrence runs per head in parallel.
- **Learnable initial weights.** The MLP's initial weights are `nn.Parameter`s
  (`mem_weights`, one `[H, in, out]` per layer, square `mem_dim`-wide hidden
  layers, Xavier init). They are the **outer-loop** params and the **chunk-0
  anchor** re-used each forward; the inner loop updates a per-sequence copy.
- **Analytic gradients (no `torch.func`).** `_mem_grads` computes the exact
  per-token MLP gradient by hand (cached layer inputs + exact erf-GELU
  derivative). `_deep_chunk_update` runs the within-chunk momentum+forget scan in
  closed form (`_deep_scan_matrix`); the cross-chunk recurrence is sequential.
- **Chunking is a parallelization knob.** `chunk_size=1` is *exact* per-token GD;
  `chunk_size>1` re-anchors the gradient at each chunk start (the Titans
  approximation), retrieval being anchor-causal at chunk granularity.
- **Keys/queries L2-normalized** (like the linear path) for stable GD, and the
  recurrence runs in fp32 for bf16/fp16 inputs (fp64 preserved).
- **Numerical-stability fix vs. the standalone kernel.** The scan matrix masks the
  strict upper triangle *in log space* (`-inf` before `exp`) instead of
  multiplying `exp(diff)` by a 0/1 mask. The standalone kernel was only ever run
  forward/inference; under training the masked `exp(diff)` can overflow to `inf`
  and `0·inf` produces `nan` in the backward pass. Log-space masking keeps both
  passes finite without changing the forward values.

The deep path reuses the shared output tail (gated RMSNorm + `o_proj`), so
`model.py` / `TitansBlock` are untouched. `mem_depth=1` (linear) is byte-for-byte
unchanged.

## Memory-module API (contract, both phases)

`NeuralMemory` (in `layers.py`) is the shared `nn.Module` both phases converge on:

```python
NeuralMemory(
    dim: int,                 # residual-stream width (in/out)
    mem_dim: int = 64,        # per-head memory (key/value) dimension
    num_heads: int | None = None,   # defaults to dim // mem_dim
    mem_depth: int = 1,       # 1 = linear matrix memory; >=2 = deep MLP memory
    chunk_size: int = 16,     # deep: test-time-GD chunk (1 = exact per-token GD)
    momentum: bool = True,    # data-dependent momentum (False => Gated DeltaNet)
    forget: bool = True,      # data-dependent decay/forget gate
    dtype: torch.dtype = torch.bfloat16,
)
forward(x: Tensor[B, S, dim]) -> retrieved: Tensor[B, S, dim]
```

Both memory kinds share this constructor signature and `forward` contract, so
`model.py` / `TitansBlock` are agnostic to `mem_depth`. For `mem_depth=1` the
memory is the linear matrix state above; for `mem_depth>=2` it is an MLP whose
weights are the memory, updated by the same momentum + forget gradient-descent
recurrence (the gradient of the associative loss `‖M(k_t) - v_t‖²` w.r.t. the MLP
weights replaces the linear `surprise_t`).

## Files

| File | Purpose |
|---|---|
| `config.py` | `TitansConfig` (`model_type="titans"`), with `momentum`/`forget`/`mem_depth`/`chunk_size`. |
| `layers.py` | `titans_delta_rule_recurrence`, `NeuralMemory`, `TitansRMSNorm`, `TitansMLP`, `TitansBlock`. |
| `model.py` | `TitansModel`, `TitansForCausalLM` (HF `PreTrainedModel` + `HFCheckpointingMixin` + `ModelCapabilities`). |
| `state_dict_adapter.py` | `TitansStateDictAdapter` — identity HF↔native remap + fp32 decay-gate contract. |
| `__init__.py` | Re-exports + registers with HF `AutoConfig`/`AutoModel`/`AutoModelForCausalLM`. |

Registered in `nemo_automodel/_transformers/registry.py`:
`MODEL_ARCH_MAPPING["TitansForCausalLM"]` and
`_CUSTOM_CONFIG_REGISTRATIONS["titans"]`.

Recipe: `examples/llm_pretrain/titans_pretrain.yaml` (builds via `from_config`).
Tests: `tests/unit_tests/models/titans/test_titans.py`.

## Environment installed (dedicated venv)

A dedicated venv was created to isolate from the shared `neural-memory/.venv`:

```bash
uv venv /home/ffrujeri/autotron/neural-memory/.venv-automodel --python 3.12
uv pip install --python .venv-automodel \
    "torch>=2.6.0" "transformers==5.8.1" "flash-linear-attention>=0.4.2" \
    einops "datasets>=4.0.0" pyyaml safetensors numpy huggingface-hub tiktoken \
    torchao pytest ruff
# automodel itself is used in-tree via PYTHONPATH=. (no editable install needed).
```

Resolved key versions: torch 2.12.1+cu130, transformers 5.8.1, fla 0.5.1,
triton 3.7.1, torchao 0.17.0. **Transformer Engine / flash-attn were NOT
installed** (heavy/optional, TE-only paths avoided per Phase-1 scope). The recipe
uses plain `torch.optim.AdamW` so it runs without TE.

## How to run

```bash
# All validation checks (build + train + reduction + adapter), GPU 0:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /home/ffrujeri/autotron/neural-memory/.venv-automodel/bin/python \
  tests/unit_tests/models/titans/test_titans.py

# Or via pytest:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /home/ffrujeri/autotron/neural-memory/.venv-automodel/bin/python -m pytest \
  tests/unit_tests/models/titans/test_titans.py -q

# Pretraining recipe (point dataset.file_pattern at real *.bin shards first):
CUDA_VISIBLE_DEVICES=0 automodel pretrain llm \
  -c examples/llm_pretrain/titans_pretrain.yaml --nproc-per-node 1
```

## Validation results (RTX 6000 Ada, GPU 0)

| Check | Result |
|---|---|
| (a) `AutoModelForCausalLM.from_config(TitansConfig(...))` builds + forward | PASS (20.76M params @ dim512/6L; tied head; `A_log` fp32) |
| (a′) `NeMoAutoModelForCausalLM.from_config(...)` builds + forward (recipe path) | PASS (resolves to `TitansForCausalLM`) |
| (b) Training loss decreases (AdamW, 30 steps, memorization task) | **5.568 → 0.143** |
| (c) Reduction — recurrence: `titans(eta=0)` vs fla `chunk_gated_delta_rule` | max\|Δ\| = **4.95e-04** (= fla chunk-vs-naive error) |
| (c) Reduction — module: `momentum=False` (fla) vs `eta→0` momentum path | max\|Δ\| = **4.05e-04** |
| Momentum actually changes output (`eta=0.5`) | Δ ≈ 0.18 (non-trivial) |
| State-dict adapter round-trip + fp32 (`A_log`/`dt_bias` → F32) | PASS |
| (d) Deep (`mem_depth=2`) build + forward + train (AdamW, 30 steps) | PASS — **5.585 → 0.407** (mem-MLP params get grads) |
| (e) Deep parity: chunked test-time GD (`chunk=1`) vs autograd per-token GD | max\|Δ\| = **3.5e-18 … 1.7e-16** (fp64) across depths {2,3} × {momentum/forget toggles} |
| (e) Deep parity: worst over chunk ∈ {1,2,4,8,16} × all configs | max\|Δ\| = **3.89e-16** (fp64) |
| Deep `chunk>1` re-anchors (output differs from `chunk=1`) | Δ > 1e-6 (non-trivial) |
| `ruff format` + `ruff check` (line-length 120) | clean |

The deep parity check is the integration's correctness proof: the chunkwise
analytic-gradient kernel reproduces an independent slow per-token reference that
takes the surprise via `torch.autograd` (sharing no code with the analytic
backward) to fp64 round-off, confirming both the hand-derived MLP gradients and
the closed-form within-chunk momentum+forget scan. (Run on an RTX 6000 Ada / GPU
0; pytest: `7 passed`.)
