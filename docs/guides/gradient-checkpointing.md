# Gradient (Activation) Checkpointing in NeMo AutoModel

Gradient checkpointing, also called _activation checkpointing_, trades a little extra compute for a **large reduction in GPU memory** by recomputing intermediate activations during the backwards pass instead of storing them.  
It is especially powerful when combined with memory-efficient loss functions (e.g., Linear-Cut Cross-Entropy) and parameter sharding using FSDP.

## Enable Gradient Checkpointing

### Configure in YAML
Add the `activation_checkpointing: true` flag under your distributed strategy.  
Example (snippet):

```yaml
# examples/llm_finetune/llama_3_2_1b_my_finetune.yaml
...

# FSDP2 (use strategy name; optional parallelism sizes)
distributed:
  strategy: fsdp2
  activation_checkpointing: true
  # dp_size: null
  # tp_size: 1
  # cp_size: 1
  ...
```

### Configure Programmatically
```python
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager

config = FSDP2Config(activation_checkpointing=True)
# device_mesh is created elsewhere (e.g. by the recipe via setup_distributed)
manager = FSDP2Manager(config, device_mesh=device_mesh, moe_mesh=moe_mesh)
model = manager.parallelize(model)
```

## Combine with Linear-Cut Cross-Entropy (LC-CE)

Linear-Cut Cross-Entropy (LC-CE) reduces the hidden-state memory required to compute the loss by calculating the softmax on the fly, thus avoiding the need to allocate memory for the logits.
It is already available using `nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy` and can be enabled in recipes by using the following:

```yaml
model:
  ...
  output_hidden_states: true

loss_fn:
  _target_: nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy
```

LC-CE and gradient checkpointing target **different memory hot-spots** (output layer vs. transformer blocks) so their benefits stack almost linearly.

## Example Memory Savings (H100-80GB, Llama-3.2-1B)
| Technique | Max GPU Mem (GB) | Δ vs Baseline |
|-----------|-----------------|---------------|
| Baseline | 53.03 | - |
| + FSDP (dp_size=8) | 47.59 | ↓ 10 % |
| + Gradient Checkpointing | 33.06 | ↓ 38 % |
| + LC-CE | 7.30 | ↓ 86 % |
| **FSDP + LC-CE + Checkpointing** | **7.30** | **↓ 86 %** |

:::{note}
- Measurements taken with local batch size = 8, sequence len = 2048, AdamW, PyTorch 2.8.
- Peak memory reported by `torch.cuda.max_memory_allocated()` averaged across DP ranks.
- Expect ±5 % variance depending on exact model, sequence length and GPU architecture.
:::

## Performance Considerations
1. **Extra compute**: Each checkpointed segment is recomputed once during the backward pass. In practice, the wall-clock overhead is ≈5-10% for transformer models.
2. **Throughput vs. Batch Size**: The goal is usually to _increase batch size_ or _sequence length_ while keeping throughput constant.

## Verify It Works
Run your training script and inspect the peak memory:
```bash

# If running on 8x GPUs
uv run torchrun --nproc-per-node=8 \
  examples/llm_finetune/finetune.py \
  -c examples/llm_finetune/llama3_2/llama_3_2_1b_my_finetune.yaml

# If running on 1x GPU
uv run examples/llm_finetune/finetune.py \
  -c examples/llm_finetune/llama3_2/llama_3_2_1b_my_finetune.yaml
```
If we run with the above settings (activation ckpt = on, lc-ce = on, fsdp = on), look for a log line similar to:
```
... | mem 7.30 GiB | ...
```