# Use Gradient (Activation) Checkpointing

Gradient checkpointing, also called _activation checkpointing_, trades a little extra compute for a **large reduction in GPU memory** by recomputing intermediate activations during the backward pass instead of storing them.  
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

For FSDP2, `activation_checkpointing` also accepts explicit policy strings:

```yaml
distributed:
  strategy: fsdp2
  activation_checkpointing: selective
```

Use `true` or `full` for full activation checkpointing. Use `selective` for PyTorch selective activation checkpointing on FSDP2 configs. Selective checkpointing saves expensive operations such as attention, collectives, and part of the matrix multiplications while recomputing cheaper operations during backward.

> **Note:** `selective` requires the FSDP2 strategy. Non-FSDP2 strategies (`ddp`, `megatron_fsdp`) raise an error when `selective` is requested. KV-sharing models (e.g., Gemma4) automatically fall back to sub-module checkpointing, because attention cannot be recomputed through the KV cache.

> **Tip:** Selective AC only speeds things up when the model's expensive operations are the ones being saved. To see the per-op save/recompute decisions for your model, set `NEMO_SELECTIVE_AC_TRACE=1`; each unique operation is logged once as `SAVE`, `RECOMPUTE`, or `ALTERNATE`. If an expensive op (e.g., an expert grouped-GEMM) shows up as `RECOMPUTE`, selective AC will not beat full checkpointing for that model.

> **Note (full vs. selective):** Selective AC saves the expensive operations (attention and part of the matmuls) and recomputes only the cheaper ones, so it does less recompute work than full AC while holding more activations in memory. Whether that nets out as faster, and at what memory cost, depends on the model, sequence length, and whether `torch.compile` is enabled, so benchmark full vs. selective for your own setup. When you do, keep the `torch.compile` setting the same on both sides (compare full and selective both compiled, or both uncompiled). `torch.compile` is a large speed lever on its own and helps both modes, so mixing it in makes it hard to tell which gain came from the AC mode.

> **Note (MoE/expert parallelism):** Selective AC is designed for dense transformers and generally does **not** help Mixture-of-Experts models with expert parallelism. In an MoE block the experts dominate the cost (they are cheap to recompute but expensive to store), and the expert-parallel dispatch/communication is opaque to the selective policy, so it is recomputed regardless. As a result, selective AC tends to add activation memory without a corresponding speedup for MoE, matching what reference implementations such as TorchTitan observe. Prefer **full** activation checkpointing (`true`/`full`) for MoE; selective remains supported for MoE and FSDP2 as an opt-in.

### Configure Programmatically

```python
from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.distributed.config import DistributedSetup

# Use activation_checkpointing="selective" for FSDP2 selective checkpointing.
distributed_setup = DistributedSetup.build(
    strategy="fsdp2",
    activation_checkpointing=True,
)

model = NeMoAutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    distributed_setup=distributed_setup,
)
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

LC-CE and gradient checkpointing target **different memory hot-spots** (output layer vs. transformer blocks), so their benefits stack almost linearly.

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
- Expect ±5 % variance depending on exact model, sequence length, and GPU architecture.
:::

## Performance Considerations
1. **Extra compute**: Each checkpointed segment is recomputed once during the backward pass. In practice, the wall-clock overhead is ≈5-10% for transformer models.
2. **Throughput vs. Batch Size**: The goal is usually to _increase batch size_ or _sequence length_ while keeping throughput constant.

## Verify It Works
Run your training script and inspect the peak memory:
```bash

# If running on 8x GPUs
automodel --nproc-per-node=8 examples/llm_finetune/llama3_2/llama_3_2_1b_my_finetune.yaml

# If running on 1x GPU
automodel examples/llm_finetune/llama3_2/llama_3_2_1b_my_finetune.yaml
```
If you run with the above settings (activation ckpt = on, lc-ce = on, fsdp = on), look for a log line similar to:
```
... | mem 7.30 GiB | ...
```
