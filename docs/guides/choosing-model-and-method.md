# Choose a Model and Method

:::{tip}
**TL;DR** -- Pick your model size by GPU VRAM, then pick SFT or LoRA based on how much memory you have left. When in doubt, start with **Qwen3-0.6B + LoRA** on a single GPU.
:::

## VRAM Rule of Thumb

| Precision | Formula | Example |
|-----------|---------|---------|
| **BF16** (default) | ~2 GB per 1B params (weights only) | 7B model ~14 GB |
| **QLoRA 4-bit** | ~0.5 GB per 1B params | 7B model ~4 GB |

:::{note}
Actual memory is higher during training due to optimizer states, activations, and gradients. The numbers above are for **weights only** -- multiply by ~3x for full SFT training memory.
:::

## Pick by GPU VRAM

| Your GPU VRAM | Max Model (SFT, BF16) | Max Model (LoRA) | Max Model (QLoRA 4-bit) | Example Config |
|---------------|------------------------|-------------------|--------------------------|----------------|
| **8 GB** (RTX 3070) | ~1B | ~3B | ~7B | `qwen/qwen3_0p6b_hellaswag.yaml` |
| **24 GB** (RTX 4090) | ~3B | ~8B | ~30B | `qwen/qwen2_5_7b_squad_peft.yaml` |
| **80 GB** (A100/H100) | ~13B | ~30B | ~70B | `llama3_3/llama_3_3_70b_instruct_squad_peft.yaml` |
| **Multi-GPU** (FSDP2) | 70B+ | 70B+ | -- | `deepseek_v32/deepseek_v32_hellaswag_pp.yaml` |

:::{tip}
All example configs live in `examples/llm_finetune/`. Run with:
```bash
automodel finetune llm -c examples/llm_finetune/<config>
```
:::

## Pick by Task

| Your Goal | Recommended Method | Why |
|-----------|-------------------|-----|
| **Domain adaptation** (medical, legal, code) | SFT or LoRA | Teaches domain vocabulary and patterns |
| **Chat / instruction following** | LoRA on an instruct model | Fast, preserves general ability |
| **Code generation** | LoRA on a code model (Qwen-Coder, Seed-Coder) | Targeted improvement |
| **Multilingual** | SFT on a multilingual base (Qwen, Gemma) | Needs more data than LoRA |
| **Image + text (VLM)** | PEFT on Gemma-3-VL or Qwen2.5-VL | Memory-efficient for large vision encoders |
| **Build from scratch** | Pretraining | See [Pretraining Guide](llm/pretraining.md) |

## SFT vs LoRA vs QLoRA

| | SFT | LoRA | QLoRA |
|---|-----|------|-------|
| **What it updates** | All weights | <1% of weights (adapters) | Same as LoRA, model quantized to 4-bit |
| **VRAM** | ~6x model size | ~2x model size | ~0.7x model size |
| **Training speed** | Fastest (per step) | Slightly slower | Slower (quantization overhead) |
| **Quality ceiling** | Highest | Very close to SFT | Slightly lower |
| **When to use** | Enough VRAM, best quality | Large model on limited VRAM | Largest models on consumer GPUs |

## Quick-Start Recommendations

- **First time?** Start with [Quickstart](quickstart.md) -- Qwen3-0.6B + HellaSwag
- **Production fine-tune?** LoRA on the largest model your GPU can fit
- **Research / maximum quality?** Full SFT with FSDP2 across multiple GPUs

:::{note}
For recommended learning rates, epochs, and LoRA rank values, see the [Hyperparameters Guide](hyperparameters.md).
:::
