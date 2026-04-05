# DeepSeek-V3

[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) is a large-scale Mixture-of-Experts model with 671B total parameters and 37B activated per token. It features Multi-head Latent Attention (MLA), innovative load balancing, and Multi-Token Prediction (MTP). DeepSeek-V3.2 is an updated release with further improvements.

[Moonlight](https://huggingface.co/moonshotai/Moonlight-16B-A3B) by Moonshot AI also uses this architecture with 16B total / 3B activated parameters.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (MoE) |
| **Architecture** | `DeepseekV3ForCausalLM` / `DeepseekV32ForCausalLM` |
| **Parameters** | 671B total / 37B active |
| **HF Org** | [deepseek-ai](https://huggingface.co/deepseek-ai) |
:::

## Available Models

- **DeepSeek-V3**: 671B total, 37B activated
- **DeepSeek-V3.2** (`DeepseekV32ForCausalLM`): updated architecture
- **Moonlight-16B-A3B** (Moonshot AI): 16B total, 3B activated

## Architectures

- `DeepseekV3ForCausalLM`
- `DeepseekV32ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| DeepSeek-V3 | [`deepseek-ai/DeepSeek-V3`](https://huggingface.co/deepseek-ai/DeepSeek-V3) |
| DeepSeek-V3-Base | [`deepseek-ai/DeepSeek-V3-Base`](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base) |
| DeepSeek-V3.2 | [`deepseek-ai/DeepSeek-V3.2`](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) |
| Moonlight 16B A3B | [`moonshotai/Moonlight-16B-A3B`](https://huggingface.co/moonshotai/Moonlight-16B-A3B) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`deepseek_v32_hellaswag_pp.yaml <../../../examples/llm_finetune/deepseek_v32/deepseek_v32_hellaswag_pp.yaml>` | SFT — DeepSeek-V3.2 on HellaSwag with pipeline parallelism |
| {download}`moonlight_16b_te.yaml <../../../examples/llm_finetune/moonlight/moonlight_16b_te.yaml>` | SFT — Moonlight 16B with Transformer Engine |
| {download}`moonlight_16b_te_packed_sequence.yaml <../../../examples/llm_finetune/moonlight/moonlight_16b_te_packed_sequence.yaml>` | SFT — Moonlight 16B with packed sequences |


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
This recipe was validated on **32 nodes × 8 GPUs (256 H100s)**. See the [Launcher Guide](../../../launcher/slurm.md) for multi-node setup.
:::

**3. Run the recipe** from inside the repo:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/deepseek_v32/deepseek_v32_hellaswag_pp.yaml
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
automodel --nproc-per-node=8 examples/llm_finetune/deepseek_v32/deepseek_v32_hellaswag_pp.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md) and the [Large MoE Fine-Tuning Guide](../../../guides/llm/large_moe_finetune.md).

## Hugging Face Model Cards

- [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- [deepseek-ai/DeepSeek-V3-Base](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base)
- [moonshotai/Moonlight-16B-A3B](https://huggingface.co/moonshotai/Moonlight-16B-A3B)
