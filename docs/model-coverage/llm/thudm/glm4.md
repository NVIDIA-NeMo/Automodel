# GLM-4

[GLM-4](https://github.com/zai-org/GLM-4) is Zhipu AI's fourth-generation General Language Model, featuring strong multilingual capabilities and tool-use support.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `GlmForCausalLM` / `Glm4ForCausalLM` |
| **Parameters** | 9B – 32B |
| **HF Org** | [zai-org](https://huggingface.co/zai-org) |
:::

## Available Models

- **GLM-4-9B-Chat-HF** (`GlmForCausalLM`): 9B
- **GLM-4-32B-0414** (`Glm4ForCausalLM`): 32B

## Architectures

- `GlmForCausalLM` — GLM-4 series
- `Glm4ForCausalLM` — GLM-4-0414 series

## Example HF Models

| Model | HF ID |
|---|---|
| GLM-4-9B-Chat-HF | [`zai-org/glm-4-9b-chat-hf`](https://huggingface.co/zai-org/glm-4-9b-chat-hf) |
| GLM-4-32B-0414 | [`zai-org/GLM-4-32B-0414`](https://huggingface.co/zai-org/GLM-4-32B-0414) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`glm_4_9b_chat_hf_squad.yaml <../../../../examples/llm_finetune/glm/glm_4_9b_chat_hf_squad.yaml>` | SFT — GLM-4 9B on SQuAD |
| {download}`glm_4_9b_chat_hf_hellaswag_fp8.yaml <../../../../examples/llm_finetune/glm/glm_4_9b_chat_hf_hellaswag_fp8.yaml>` | SFT — GLM-4 9B on HellaSwag with FP8 |


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
automodel --nproc-per-node=8 examples/llm_finetune/glm/glm_4_9b_chat_hf_squad.yaml
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
automodel --nproc-per-node=8 examples/llm_finetune/glm/glm_4_9b_chat_hf_squad.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [zai-org/glm-4-9b-chat-hf](https://huggingface.co/zai-org/glm-4-9b-chat-hf)
- [zai-org/GLM-4-32B-0414](https://huggingface.co/zai-org/GLM-4-32B-0414)
