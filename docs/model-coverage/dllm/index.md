# Diffusion Language Models (dLLMs)

Diffusion language models (dLLMs) generate text by **denoising** rather than left-to-right autoregression. A fixed-length response "canvas" is corrupted and then iteratively refined, so tokens are produced in parallel and can be revised across steps. NeMo AutoModel supports fine-tuning block-diffusion dLLMs with the same recipe-driven, FSDP2/Expert-Parallel training stack used for LLMs and VLMs.

## Supported Models

| Owner | Model Family | Architectures |
|---|---|---|
| Google | [DiffusionGemma](google/diffusiongemma.md) | `DiffusionGemmaForBlockDiffusion` |

## Fine-Tuning

See the [DiffusionGemma Fine-Tuning Guide](../../guides/dllm/diffusiongemma.md) for the block-diffusion training objective (uniform-random token corruption, no `[MASK]`), self-conditioning, and the supported feature set (SFT, LoRA, Expert Parallelism, activation checkpointing).

```{toctree}
:hidden:

google/diffusiongemma
```
