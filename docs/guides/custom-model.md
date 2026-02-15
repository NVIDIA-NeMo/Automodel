# Train a Custom Model

:::{tip}
**TL;DR** -- Bring your own `nn.Module` and train it with AutoModel's distributed recipes. Three paths depending on your needs.
:::

## Three Paths

- **Path A: `_target_` any `nn.Module`** (simplest) -- Point `_target_` in your YAML at any callable that returns an `nn.Module`. No registration needed.
- **Path B: Model registry** -- Register your model class so `from_pretrained` dispatches to it automatically.
- **Path C: Functional pipeline API** -- For pipeline parallelism with custom models that need explicit stage splitting.

## Path A: Point `_target_` at Your Model (Simplest)

Write a builder function that returns an `nn.Module`:

```python
# my_models/tiny_transformer.py
import torch.nn as nn

def build_my_model(hidden_size=256, num_layers=4, vocab_size=32000):
    """Returns a standard nn.Module. AutoModel handles the rest."""
    config = {"hidden_size": hidden_size, "num_layers": num_layers, "vocab_size": vocab_size}
    # ... build your model here ...
    return model
```

Reference it in YAML:

```yaml
model:
  _target_: my_models.tiny_transformer.build_my_model  # any Python import path
  hidden_size: 256
  num_layers: 4
  vocab_size: 32000
```

Run:

```bash
automodel finetune llm -c my_config.yaml
```

:::{tip}
`_target_` can also point to a local file: `_target_: path/to/file.py:build_my_model`. See [YAML Configuration](configuration.md) for details.
:::

## Path B: Register in the Model Registry

If you want your model to be discoverable by `NeMoAutoModelForCausalLM.from_pretrained()`:

```python
from nemo_automodel._transformers.registry import register_modeling_path

# Register a module path that contains your model class
register_modeling_path("my_models.custom_llm")
```

Your model class must follow the HF naming convention: the class name should match the `model_type` in your config (e.g., `MyCustomForCausalLM` for `model_type: "my_custom"`).

## Path C: Functional Pipeline API

For pipeline parallelism, your model needs explicit stage boundaries. See the [Pipelining Guide](pipelining.md) for details on defining `split_points`.

## Minimal Working Example

Here is a complete custom model based on the built-in GPT-2 template:

```yaml
# my_gpt2_config.yaml
model:
  _target_: nemo_automodel.components.models.gpt2.build_gpt2_model
  n_layer: 12
  n_embd: 768
  n_head: 12
  vocab_size: 50257

dataset:
  _target_: nemo_automodel.components.datasets.llm.hellaswag.HellaSwag
  path_or_dataset: rowan/hellaswag
  split: train

step_scheduler:
  global_batch_size: 64
  local_batch_size: 8
  num_epochs: 1

distributed:
  strategy: fsdp2
  tp_size: 1

optimizer:
  _target_: torch.optim.Adam
  lr: 1e-4
```

```bash
automodel finetune llm -c my_gpt2_config.yaml
```

## FSDP2 Compatibility Checklist

Make sure your custom `nn.Module`:

- **Uses standard PyTorch layers** (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, etc.)
- **Has no in-place operations** on parameters during forward pass
- **Returns a dict or dataclass** from `forward()` with a `loss` key (or configure a custom loss function)
- **Does not store intermediate state** across forward calls (breaks gradient checkpointing)
- **Has a `config` attribute** if using the model registry path (can be a simple dataclass or dict)

## Reference

- **GPT-2 template**: [`nemo_automodel/components/models/gpt2.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/models/gpt2.py) -- full working example
- **Model registry**: [`nemo_automodel/_transformers/registry.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/_transformers/registry.py) -- registration API
- **`_target_` syntax**: [YAML Configuration](configuration.md)
