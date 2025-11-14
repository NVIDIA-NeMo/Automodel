# Common Model Components

This directory contains **shared, reusable components** for custom transformer model implementations in NeMo Automodel.

## Architecture Overview

Instead of duplicating code across different model implementations (Llama, Qwen2, etc.), we provide modular, composable components that can be mixed into any transformer architecture.

## Components

### 1. `combined_projection/combined_qkv.py` - CombinedQKVAttentionMixin

**Purpose**: Provides efficient combined QKV projection for attention modules.

**Benefits**:
- **Memory efficiency**: Single projection reduces memory footprint
- **Performance**: Fewer kernel launches, better memory coalescing
- **Simplicity**: One code path, always optimized

**Usage**:
```python
from nemo_automodel.components.models.common.combined_projection import CombinedQKVAttentionMixin

class MyAttention(CombinedQKVAttentionMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        # Setup combined QKV projection (ALWAYS combined in custom implementation)
        self.setup_qkv_projection(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=self.head_dim,
            bias=config.attention_bias,
            use_combined_qkv=True,  # Always True in custom implementations
        )
    
    def forward(self, hidden_states, ...):
        # Compute Q, K, V from combined projection
        q, k, v = self.compute_qkv(hidden_states)
        # ... rest of attention logic ...
```

**Key Features**:
- Automatic tensor parallelism support (dynamic split sizing)
- State dict adapter handles loading HuggingFace checkpoints
- ALWAYS uses combined projections - this is the whole point of custom implementations

### 2. `combined_projection/combined_mlp.py` - CombinedGateUpMLP

**Purpose**: Combines gate_proj and up_proj into a single efficient projection for SwiGLU-style MLPs.

**Benefits**:
- **Reduced overhead**: Single projection + split vs two separate projections
- **Cleaner code**: One component instead of duplicated implementations
- **Universal**: Works with any SwiGLU variant (Llama, Qwen2, etc.)

**Usage**:
```python
from nemo_automodel.components.models.common.combined_projection import CombinedGateUpMLP

# For any SwiGLU-style MLP (Llama, Qwen2, etc.)
mlp = CombinedGateUpMLP(config)  # config.hidden_act = "silu"
output = mlp(hidden_states)
```

**Key Features**:
- Automatic activation function selection from config
- Tensor parallelism support
- Compatible with any gate * up activation pattern

### 3. `state_dict_adapter.py` - CombinedProjectionStateDictAdapter

**Purpose**: Generic converter between HuggingFace format (separate projections) and custom format (combined projections).

**Benefits**:
- **Universal**: Works with Llama, Qwen2, and any similar architecture
- **DRY principle**: No duplicated conversion logic
- **Safe**: Handles DTensor sharding without OOM

**Usage**:
```python
from nemo_automodel.components.models.common.state_dict_adapter import CombinedProjectionStateDictAdapter

# Works for any model with Q/K/V and gate/up projections
adapter = CombinedProjectionStateDictAdapter(config)

# Convert HF checkpoint to custom format
custom_state_dict = adapter.from_hf(hf_state_dict)

# Convert custom checkpoint back to HF format
hf_state_dict = adapter.to_hf(custom_state_dict)
```

**Key Features**:
- Automatic detection of model prefix (`model.` or not)
- DTensor-aware (no all-gather OOM)
- Handles both weights and biases
- Regex-based key exclusion support

## Model-Specific Implementations

### Llama (`../llama/`)

```python
from nemo_automodel.components.models.llama import build_llama_model

# Load with combined projections
model = build_llama_model(
    "meta-llama/Llama-3-8B",
    use_combined_qkv=True,
    use_combined_gate_up=True
)
```

**State Dict Adapter**:
```python
from nemo_automodel.components.models.llama.state_dict_adapter import LlamaStateDictAdapter
```

### Qwen2 (`../qwen2/`)

```python
from nemo_automodel.components.models.qwen2 import build_qwen2_model

# Load with combined projections
model = build_qwen2_model(
    "Qwen/Qwen2.5-7B",
    use_combined_qkv=True,
    use_combined_gate_up=True
)
```

**State Dict Adapter**:
```python
from nemo_automodel.components.models.qwen2.state_dict_adapter import Qwen2StateDictAdapter
```

## Design Principles

### 1. **Composition over Inheritance**
   - Mixin classes for flexible composition
   - Reusable components that don't force inheritance hierarchies

### 2. **Zero-Copy Semantics**
   - No unnecessary tensor copies or all-gathers
   - DTensor operations stay local

### 3. **Graceful Degradation**
   - `use_fused_qkv=False` falls back to separate projections
   - Compatible with existing checkpoints via state dict adapters

### 4. **Configuration-Driven**
   - All options controllable via config/YAML
   - No hardcoded assumptions about model architecture

## Adding Support for New Models

To add combined projection support for a new transformer model:

1. **Attention**: Use `CombinedQKVAttentionMixin`
   ```python
   from nemo_automodel.components.models.common.combined_projection import CombinedQKVAttentionMixin
   
   class NewModelAttention(CombinedQKVAttentionMixin, nn.Module):
       def __init__(self, config, layer_idx, use_combined_qkv=False):
           super().__init__()
           self.setup_qkv_projection(...)  # Setup from mixin
       
       def forward(self, hidden_states, ...):
           q, k, v = self.compute_qkv(hidden_states)  # Use mixin
           # ... rest of attention ...
   ```

2. **MLP**: Use `CombinedGateUpMLP` (if SwiGLU-style)
   ```python
   from nemo_automodel.components.models.common.combined_projection import CombinedGateUpMLP
   
   if use_combined_gate_up:
       self.mlp = CombinedGateUpMLP(config)
   else:
       self.mlp = StandardMLP(config)
   ```

3. **State Dict**: Subclass `CombinedProjectionStateDictAdapter`
   ```python
   class NewModelStateDictAdapter(CombinedProjectionStateDictAdapter):
       def __init__(self, config: NewModelConfig):
           super().__init__(config)
   ```

That's it! No need to reimplement conversion logic.

## Tensor Parallelism Support

All components automatically handle tensor parallelism:

- **Dynamic split sizing**: Computes local sizes based on actual tensor dimensions
- **No redistribution**: DTensor operations stay on local shards
- **Transparent**: Works the same whether sharded or not

Example with TP=4:
```python
# config.tp_size = 4
model = build_qwen2_model("Qwen/Qwen2.5-7B", use_combined_qkv=True)
# QKV projection is automatically sharded 4-way
# Splits computed dynamically based on local shard size
```

## Performance Considerations

**When to use combined projections**:
- ✅ Large hidden sizes (memory bound)
- ✅ High tensor parallelism degree
- ✅ Memory-constrained environments

**When separate might be better**:
- ❌ Very small models (overhead dominates)
- ❌ Specific hardware optimizations for separate projections
- ❌ Debugging (easier to inspect separate weights)

## Testing

All components include:
- Unit tests for correctness
- TP sharding tests
- Checkpoint conversion tests
- Numerical equivalence tests vs HuggingFace

Run tests:
```bash
pytest tests/components/models/common/
```

## References

- **Llama**: Meta's Llama 2/3 architecture
- **Qwen2**: Qwen 2.5 architecture with sliding window attention
- **Tensor Parallelism**: PyTorch DTensor (FSDP2-compatible)
- **HuggingFace Transformers**: Baseline implementations

