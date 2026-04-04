# Adding Model Support to NeMo AutoModel

This skill guides implementation of new model architectures in NeMo AutoModel. Follow the five phases in order.

## Phase 1: Discovery

Before writing code, gather information about the target model.

### 1.1 Fetch HuggingFace config.json

Download the model's `config.json` from the HuggingFace Hub (or use `AutoConfig.from_pretrained`). Key fields to extract:

- `architectures` -- determines the class name and registration key (e.g., `"LlamaForCausalLM"`, `"Qwen3MoeForCausalLM"`, `"Mistral3ForConditionalGeneration"`)
- `model_type` -- used for custom config registration in `_CUSTOM_CONFIG_REGISTRATIONS` if HF does not have a built-in config class
- `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads` -- sizing
- `vocab_size` -- needed for tiny test configs
- `tie_word_embeddings` -- whether lm_head shares weights with embed_tokens
- `hidden_act` -- activation function (e.g., `"silu"` for SwiGLU)

### 1.2 Determine model type

| Type | Indicators | Pattern file |
|------|-----------|-------------|
| **Dense LLM** | `ForCausalLM` in architectures, no expert fields | [llm-patterns.md](./llm-patterns.md) |
| **MoE LLM** | `n_routed_experts`, `num_local_experts`, `num_experts_per_tok` in config | [moe-patterns.md](./moe-patterns.md) |
| **VLM** | `ForConditionalGeneration` in architectures, has `vision_config` + `text_config` | [vlm-patterns.md](./vlm-patterns.md) |

### 1.3 Check for existing similar architectures

Look in `components/models/` for architectures with similar attention or MLP patterns:

```
components/models/
  llama/           # Standard GQA + SwiGLU (CombinedQKV + CombinedGateUpMLP)
  qwen2/           # Same as Llama but with attention bias + QKV bias
  baichuan/        # ALiBi attention variant
  deepseek_v3/     # MLA attention + MoE (DeepSeek-style grouped experts)
  mistral4/        # MLA + MoE + VLM (Pixtral vision)
  kimivl/          # DeepSeek-V3 backbone + MoonVit vision
  kimi_k25_vl/     # Updated KimiVL with different projector
  qwen3_moe/       # Qwen3 with MoE layers
  nemotron_v3/     # Hybrid mamba-attention
```

### 1.4 Identify custom components

Check whether the model needs:

- **Custom attention**: GQA (standard), MLA (DeepSeek/Mistral4), sliding window, bidirectional
- **Custom RoPE**: Standard (Llama), YaRN scaling, NTK-aware, complex-number (DeepSeek)
- **Custom normalization**: RMSNorm (standard), LayerNorm, different eps values
- **Custom MLP**: SwiGLU (standard), GeGLU, ReLU-squared, MoE routing
- **Custom config class**: Needed only if HF `AutoConfig` cannot parse the model's `config.json` (check `auto_map` field)

### 1.5 Note dimensions for test config

For unit tests, create a tiny config. Target: ~1M parameters or less.

```python
# Example tiny config for a Llama-like model:
tiny_config = LlamaConfig(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=256,
    max_position_embeddings=128,
)
```

---

## Phase 2: Implementation

### 2.1 Create directory structure

```
components/models/<name>/
  __init__.py
  model.py
  state_dict_adapter.py
  config.py            # Only if HF config is insufficient
  layers.py            # Only for MoE / MLA / other non-standard layers
  rope_utils.py        # Only for custom RoPE
```

### 2.2 Implementation order

Implement files in dependency order:

1. **config.py** (if needed) -- Custom `PretrainedConfig` subclass
2. **rope_utils.py** (if needed) -- RoPE implementation
3. **layers.py** (if needed) -- Attention, MLP, decoder block classes
4. **model.py** -- The main `ForCausalLM` (or `ForConditionalGeneration`) class
5. **state_dict_adapter.py** -- HF weight conversion
6. **__init__.py** -- Re-export the main model class

See the pattern files for detailed implementation guidance:

- Dense LLM: [llm-patterns.md](./llm-patterns.md)
- MoE: [moe-patterns.md](./moe-patterns.md)
- VLM: [vlm-patterns.md](./vlm-patterns.md)

### 2.3 Register in registry

Add the model to `MODEL_ARCH_MAPPING` in `_transformers/registry.py`:

```python
# In _transformers/registry.py
MODEL_ARCH_MAPPING = OrderedDict([
    # ... existing entries ...
    (
        "NewModelForCausalLM",
        ("nemo_automodel.components.models.new_model.model", "NewModelForCausalLM"),
    ),
])
```

If the model has a custom config class with `auto_map` in its `config.json`, also register in `_CUSTOM_CONFIG_REGISTRATIONS`:

```python
_CUSTOM_CONFIG_REGISTRATIONS: Dict[str, Tuple[str, str]] = {
    # ... existing entries ...
    "new_model": ("nemo_automodel.components.models.new_model.configuration", "NewModelConfig"),
}
```

---

## Phase 3: Recipe & Config

### 3.1 Create example YAML config

Create an example config under `examples/llm_finetune/<name>/` (or `examples/vlm_finetune/<name>/`):

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: <org>/<model-name>

trainer:
  max_steps: 100
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1

# ... data, optimizer config ...
```

### 3.2 Verify model loads

Test that the model loads from a HuggingFace checkpoint:

```python
from nemo_automodel import NeMoAutoModelForCausalLM

model = NeMoAutoModelForCausalLM.from_pretrained("<org>/<model-name>")
```

### 3.3 Test with tiny config first

Before using full-size models, verify with a tiny config (1-2 layers, small hidden dim) to catch shape mismatches early.

---

## Phase 4: Tests

### General testing rules

- **Match the model's dtype.** Never invent or override the dtype in tests. If the
  model runs in `bfloat16`, tests must also use `bfloat16`. Check the model's
  `config.json` (`torch_dtype` field) or its default `model.dtype` and use that
  everywhere -- model init, input tensors, reference tensors, and tolerance
  thresholds. Using a different dtype (e.g., `float32` when the model uses
  `bfloat16`) hides real precision issues and makes parity comparisons meaningless.

- **Layer-rewrite equivalence tests.** Whenever you rewrite or replace a layer
  (attention, MLP, normalization, RoPE, etc.) with a custom implementation, you
  **must** add a unit test that checks numerical equivalence against the original
  HuggingFace layer. Guidelines:
  - Use a **realistically sized** layer config, not the absolute minimum. For
    example, `hidden_size=256, num_attention_heads=8, num_key_value_heads=4` is
    a reasonable choice. Extremely small dimensions (e.g., `hidden_size=8`) can
    mask numerical divergence because there are too few elements for errors to
    accumulate.
  - Seed both the original and rewritten layer with identical weights (copy
    `state_dict` from one to the other via the adapter if needed).
  - Feed the same random input through both layers and assert
    `torch.allclose(out_original, out_rewritten, atol=..., rtol=...)` with
    tolerances appropriate for the dtype (`atol=1e-2, rtol=1e-2` for `bfloat16`;
    `atol=1e-5, rtol=1e-5` for `float32`).
  - Name the test `test_<layer>_equivalence` (e.g., `test_attention_equivalence`,
    `test_mlp_equivalence`).

### 4.1 Unit tests

Create `tests/unit_tests/models/<name>/` with:

```python
import pytest
import torch
from transformers import AutoConfig  # or custom config

@pytest.fixture
def tiny_config():
    return AutoConfig.from_pretrained(...)  # or build manually with small dims

@pytest.fixture
def model(tiny_config):
    from nemo_automodel.components.models.<name>.model import <Name>ForCausalLM
    return <Name>ForCausalLM(tiny_config)

def test_forward_shape(model, tiny_config):
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))
    output = model(input_ids)
    # Check output shape matches expectations
    assert output.logits.shape == (batch_size, seq_len, tiny_config.vocab_size)

def test_state_dict_roundtrip(model):
    """Verify from_hf -> to_hf round-trip preserves weights."""
    adapter = model.state_dict_adapter
    original_sd = model.state_dict()
    hf_sd = adapter.to_hf(original_sd)
    restored_sd = adapter.from_hf(hf_sd)
    for key in original_sd:
        assert torch.allclose(original_sd[key], restored_sd[key]), f"Mismatch at {key}"
```

### 4.2 Layer equivalence tests

When a layer has been rewritten, add a test like:

```python
def test_attention_equivalence(tiny_config):
    """Verify custom attention matches HF reference output."""
    dtype = tiny_config.torch_dtype  # e.g. torch.bfloat16
    torch.manual_seed(42)

    # Instantiate original HF layer and custom NeMo layer
    from transformers.models.<name>.modeling_<name> import <Name>Attention as HFAttention
    from nemo_automodel.components.models.<name>.layers import <Name>Attention as NeMoAttention

    hf_attn = HFAttention(tiny_config, layer_idx=0).to(dtype=dtype, device="cuda")
    nemo_attn = NeMoAttention(tiny_config, layer_idx=0).to(dtype=dtype, device="cuda")

    # Copy weights from HF to NeMo (via state_dict adapter or manual mapping)
    # ... adapter.from_hf(hf_attn.state_dict()) ...

    hidden = torch.randn(2, 32, tiny_config.hidden_size, dtype=dtype, device="cuda")
    position_ids = torch.arange(32, device="cuda").unsqueeze(0).expand(2, -1)

    with torch.no_grad():
        hf_out = hf_attn(hidden, position_ids=position_ids)[0]
        nemo_out = nemo_attn(hidden, position_ids=position_ids)[0]

    atol, rtol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-5, 1e-5)
    assert torch.allclose(hf_out, nemo_out, atol=atol, rtol=rtol), (
        f"Max diff: {(hf_out - nemo_out).abs().max().item()}"
    )
```

### 4.3 Functional tests

Create a short training test (few steps) that verifies loss decreases:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_training_loss_decreases(model, tiny_config):
    # ... setup optimizer, dummy data ...
    losses = []
    for step in range(5):
        loss = model(input_ids, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < losses[0], "Loss should decrease during training"
```

---

## Phase 5: Documentation

### 5.1 Update model coverage page

Edit the appropriate file in `docs/model-coverage/`:
- LLM/MoE: `docs/model-coverage/llm.md`
- VLM: `docs/model-coverage/vlm.md`

Add a row with the model name, supported features (TP, PP, FSDP, LoRA, QLoRA), and any limitations.

### 5.2 Add example configs

Add fully commented example configs under `examples/`:
- `examples/llm_finetune/<name>/sft.yaml` for LLMs
- `examples/vlm_finetune/<name>/sft.yaml` for VLMs

---

## Phase 6: Parity Testing

After implementation and unit tests are complete, run the full parity-testing
workflow to verify that the new model produces numerically equivalent results to
the reference HuggingFace implementation.

**Read and follow the parity-testing skill** at
`.claude/skills/parity-testing/SKILL.md`. It walks through three levels of
comparison (state-dict round-trip, component-level parity, end-to-end forward
pass) and provides debugging steps when a level fails.

Do not skip this phase. A model that passes unit tests can still diverge from HF
due to subtle weight-conversion bugs, backend differences, or RoPE mismatches
that only surface in a full parity comparison.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `_transformers/registry.py` | `MODEL_ARCH_MAPPING` and `_CUSTOM_CONFIG_REGISTRATIONS` |
| `components/models/common/__init__.py` | Exports `CombinedQKVAttentionMixin`, `CombinedGateUpMLP`, `BackendConfig`, `HFCheckpointingMixin`, etc. |
| `components/models/common/combined_projection/combined_qkv.py` | `CombinedQKVAttentionMixin` with `setup_qkv_projection()` and `compute_qkv()` |
| `components/models/common/combined_projection/combined_mlp.py` | `CombinedGateUpMLP` with interleaved gate/up layout |
| `components/models/common/combined_projection/state_dict_adapter.py` | `CombinedProjectionStateDictAdapter` base class |
| `components/models/common/hf_checkpointing_mixin.py` | `HFCheckpointingMixin` for save/load |
| `components/models/common/utils.py` | `BackendConfig`, `initialize_rms_norm_module`, `initialize_linear_module`, `get_rope_config` |
| `components/moe/config.py` | `MoEConfig` dataclass |
| `components/moe/fsdp_mixin.py` | `MoEFSDPSyncMixin` for distributed expert handling |
| `components/moe/layers.py` | `MoE` layer, `MLP` (dense) for MoE blocks |
| `components/moe/experts.py` | `GroupedExperts`, `GroupedExpertsDeepEP`, `GroupedExpertsTE` |

---

## Checklist

- [ ] Fetched and analyzed `config.json` from HuggingFace
- [ ] Determined model type (dense LLM / MoE / VLM)
- [ ] Identified custom components (attention, RoPE, normalization, MLP)
- [ ] Created `components/models/<name>/` directory
- [ ] Implemented config.py (if custom config needed)
- [ ] Implemented layers.py (if custom layers needed)
- [ ] Implemented rope_utils.py (if custom RoPE needed)
- [ ] Implemented model.py with `HFCheckpointingMixin`
- [ ] Implemented state_dict_adapter.py
- [ ] Implemented __init__.py with re-export
- [ ] Registered in `MODEL_ARCH_MAPPING` in `_transformers/registry.py`
- [ ] Registered custom config in `_CUSTOM_CONFIG_REGISTRATIONS` (if applicable)
- [ ] Created example YAML config
- [ ] Verified model loads via `NeMoAutoModelForCausalLM.from_pretrained()`
- [ ] Created unit tests (forward shape, state_dict round-trip)
- [ ] Created layer equivalence tests for every rewritten layer (matching model dtype)
- [ ] Created functional tests (training loss decreases)
- [ ] Updated docs/model-coverage page
- [ ] Ran parity-testing skill (state-dict round-trip, component parity, E2E forward pass)
- [ ] Set `ModelClass = <Name>ForCausalLM` at module bottom
