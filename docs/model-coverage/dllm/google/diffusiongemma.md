# DiffusionGemma

[DiffusionGemma](https://huggingface.co/google) is a block-diffusion language model from Google. Instead of generating tokens left-to-right, it denoises a fixed-length canvas of tokens in parallel: a causal encoder reads the prompt and a bidirectional decoder iteratively refines the response canvas. The released checkpoint is a Mixture-of-Experts model with 26B total parameters and ~4B active per token.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (Block Diffusion, MoE) |
| **Architecture** | `DiffusionGemmaForBlockDiffusion` |
| **Parameters** | 26B total / ~4B active |
| **HF Org** | [google](https://huggingface.co/google) |
:::

## Available Models

- **DiffusionGemma 26B-A4B-it** (`DiffusionGemmaForBlockDiffusion`): instruction-tuned block-diffusion MoE.

## Architectures

- `DiffusionGemmaForBlockDiffusion` — block-diffusion MoE (causal prompt encoder + bidirectional canvas decoder).

## Example HF Models

| Model | HF ID |
|---|---|
| DiffusionGemma 26B-A4B-it | [`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`diffusion_gemma_sft.yaml <../../../../examples/dllm_sft/diffusion_gemma_sft.yaml>` | Full SFT — DiffusionGemma 26B-A4B with FSDP2 + Expert Parallelism |
| {download}`diffusion_gemma_lora.yaml <../../../../examples/dllm_sft/diffusion_gemma_lora.yaml>` | LoRA SFT — DiffusionGemma 26B-A4B with FSDP2 + Expert Parallelism |

## Try with NeMo AutoModel

**1. Install** ([full instructions](../../../guides/installation.md)):

```bash
pip install nemo-automodel
```

**2. Clone the repo** to get the example recipes:

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
```

:::{note}
This recipe was validated with **Expert Parallelism (EP=8)** on a single 8×H100 node. See the [Launcher Guide](../../../launcher/slurm.md) for multi-node setup.
:::

**3. Run the recipe** from inside the repo:

```bash
automodel --nproc-per-node=8 examples/dllm_sft/diffusion_gemma_sft.yaml
```

## Fine-Tuning

See the [DiffusionGemma Fine-Tuning Guide](../../../guides/dllm/diffusiongemma.md) for the block-diffusion training objective, self-conditioning, and the full list of supported features (SFT, LoRA, Expert Parallelism, activation checkpointing).

## Hugging Face Model Cards

- [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
