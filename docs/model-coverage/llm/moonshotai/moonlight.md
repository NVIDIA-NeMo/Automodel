# Moonlight

[Moonlight](https://huggingface.co/moonshotai/Moonlight-16B-A3B) is a Mixture-of-Experts language model from Moonshot AI trained using Muon optimizer. It uses the `DeepseekV3ForCausalLM` architecture with 16B total parameters and 3B activated per token.

:::{card}
| | |
|---|---|
| **Task** | Text Generation (MoE) |
| **Architecture** | `DeepseekV3ForCausalLM` |
| **Parameters** | 16B total / 3B active |
| **HF Org** | [moonshotai](https://huggingface.co/moonshotai) |
:::

## Available Models

- **Moonlight-16B-A3B**: 16B total, 3B activated

## Architecture

- `DeepseekV3ForCausalLM` (same architecture as DeepSeek-V3)

## Example HF Models

| Model | HF ID |
|---|---|
| Moonlight 16B A3B | [`moonshotai/Moonlight-16B-A3B`](https://huggingface.co/moonshotai/Moonlight-16B-A3B) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`moonlight_16b_te.yaml <../../../../examples/llm_finetune/moonlight/moonlight_16b_te.yaml>` | SFT — Moonlight 16B with Transformer Engine |
| {download}`moonlight_16b_te_packed_sequence.yaml <../../../../examples/llm_finetune/moonlight/moonlight_16b_te_packed_sequence.yaml>` | SFT — Moonlight 16B with packed sequences |


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

**3. Run the recipe** from inside the repo:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/moonlight/moonlight_16b_te.yaml
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
automodel --nproc-per-node=8 examples/llm_finetune/moonlight/moonlight_16b_te.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [moonshotai/Moonlight-16B-A3B](https://huggingface.co/moonshotai/Moonlight-16B-A3B)
