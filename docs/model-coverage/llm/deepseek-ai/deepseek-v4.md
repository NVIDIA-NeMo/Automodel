# DeepSeek-V4-Flash

[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) is a 284B-total / 13B-active Mixture-of-Experts model with a 1M-token context. It introduces:

- **Hyper-Connections (mHC)** — the hidden state is kept as ``[B, S, hc_mult, dim]`` between blocks, with a learned Sinkhorn-mixed residual at every attention and FFN site.
- **Compressed Sparse Attention (CSA)** and **Heavily Compressed Attention (HCA)** — per-layer ``compress_ratio ∈ {0, 4, 128}`` with a learned gated pool plus a per-head top-k indexer.
- **Hash routing** for the first ``num_hash_layers`` layers (token-id → expert-id table); ``sqrt(softplus(·))`` topk routing for the remaining layers.
- **Dual RoPE bases** — ``rope_theta=10000`` for sliding-window layers, ``compress_rope_theta=160000`` (with YaRN) for compressed-KV layers.
- **Per-head learnable attention sinks** consumed by ``eager_attention_with_sink``.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (MoE) |
| **Architecture** | `DeepseekV4ForCausalLM` |
| **Parameters** | 284B total / 13B active |
| **Context** | 1M tokens |
| **HF Org** | [deepseek-ai](https://huggingface.co/deepseek-ai) |
:::

## Available Models

- **DeepSeek-V4-Flash** (`DeepseekV4ForCausalLM`): 284B total, 13B activated, 1M context.

## Architectures

- `DeepseekV4ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| DeepSeek-V4-Flash | [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`deepseek_v4_flash_hellaswag.yaml <../../../../examples/llm_finetune/deepseek_v4/deepseek_v4_flash_hellaswag.yaml>` | SFT — DeepSeek-V4-Flash on HellaSwag (PP=4, EP=32, 16-node) |


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
This recipe was validated on **16 nodes × 8 H100 (128 GPUs)** with PP=4 and EP=32. See the [Launcher Guide](../../../launcher/slurm.md) for multi-node setup.
:::

**3. Run the recipe** from inside the repo:

```bash
torchrun --nproc-per-node 8 examples/llm_finetune/deepseek_v4/deepseek_v4_flash_hellaswag.yaml
```

:::{dropdown} Run with Docker
**1. Pull the container** and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```

**2.** Navigate to the AutoModel directory (where the recipes are):

```bash
cd /opt/Automodel
```

**3. Run the recipe**:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/deepseek_v4/deepseek_v4_flash_hellaswag.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md) and the [Large MoE Fine-Tuning Guide](../../../guides/llm/large-moe-finetune.md).

## Hugging Face Model Cards

- [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
