# Hyperparameters Guide

:::{tip}
**TL;DR** -- Start with the defaults below. Adjust learning rate first if loss doesn't converge. Reduce batch size if you hit OOM.
:::

## Recommended Defaults by Task

| Parameter | SFT | LoRA | QLoRA | Pretraining |
|-----------|-----|------|-------|-------------|
| **Learning rate** | `1e-5` | `2e-4` | `2e-4` | `6e-4` |
| **LR schedule** | cosine | cosine | cosine | cosine + warmup |
| **Warmup steps** | 10 | 10 | 10 | 700 |
| **Epochs** | 1-3 | 1-3 | 1-3 | 1 |
| **Global batch size** | 64-128 | 32-64 | 32-64 | 512+ |
| **Local batch size** | 4-8 | 4-8 | 4-8 | 8-16 |
| **Weight decay** | `0.01` | `0.0` | `0.0` | `0.01` |
| **LoRA rank** | -- | 8-16 | 16 | -- |
| **LoRA alpha** | -- | 32 | 32 | -- |
| **Precision** | BF16 | BF16 | NF4 + BF16 | BF16 |

## How to Override from CLI

```bash
# Change learning rate
automodel finetune llm -c config.yaml --optimizer.lr 5e-6

# Change batch size
automodel finetune llm -c config.yaml --step_scheduler.global_batch_size 128

# Change number of epochs
automodel finetune llm -c config.yaml --step_scheduler.num_epochs 3

# Change LoRA rank and alpha
automodel finetune llm -c config.yaml --peft.lora_rank 16 --peft.lora_alpha 32
```

## Common Mistakes

:::{warning}
**Learning rate too high** -- Loss spikes or diverges after a few steps. Fix: reduce by 5-10x (e.g., `2e-4` -> `2e-5`).
:::

:::{warning}
**Too many epochs** -- Validation loss starts increasing while training loss keeps dropping (overfitting). Fix: use 1-3 epochs, add early stopping, or increase dataset size.
:::

:::{warning}
**Batch size too large for VRAM** -- CUDA OOM error. Fix: reduce `local_batch_size` and increase `gradient_accumulation_steps` to keep the same effective global batch size.
:::

:::{warning}
**LoRA rank too low** -- Model underfits (loss plateaus high). Fix: increase rank from 8 to 16 or 32. Higher rank = more parameters = more capacity but more memory.
:::

## Tuning Order

1. **Start with defaults** from the table above
2. **Adjust learning rate** -- this has the biggest impact; try 3 values (e.g., `1e-5`, `5e-5`, `2e-4`)
3. **Adjust epochs** -- monitor validation loss, stop when it starts rising
4. **Adjust batch size** -- larger batches smooth gradients but use more memory
5. **Adjust LoRA rank** (if using PEFT) -- higher rank for complex tasks, lower for simple ones

:::{note}
For model and method selection guidance, see [Choose a Model and Method](choosing-model-and-method.md).
:::
