# Ministral3 / Devstral

[Ministral](https://mistral.ai/news/ministraux/) is Mistral AI's efficient small model series optimized for on-device and edge use cases. [Devstral](https://mistral.ai/news/devstral/) is a code-focused model built on the same architecture, designed for software engineering agents.

Both use the `Mistral3ForConditionalGeneration` architecture.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `Mistral3ForConditionalGeneration` |
| **Parameters** | 3B – 24B |
| **HF Org** | [mistralai](https://huggingface.co/mistralai) |
:::

## Available Models

**Ministral3:**
- **Ministral-3-3B-Instruct-2512**
- **Ministral-3-8B-Instruct-2512**
- **Ministral-3-14B-Instruct-2512**

**Devstral:**
- **Devstral-Small-2-24B-Instruct-2512**

## Architecture

- `Mistral3ForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| Ministral-3 3B Instruct | [`mistralai/Ministral-3-3B-Instruct-2512`](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) |
| Ministral-3 8B Instruct | [`mistralai/Ministral-3-8B-Instruct-2512`](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512) |
| Ministral-3 14B Instruct | [`mistralai/Ministral-3-14B-Instruct-2512`](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512) |
| Devstral Small 2 24B | [`mistralai/Devstral-Small-2-24B-Instruct-2512`](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`devstral2_small_2512_squad.yaml <../../../examples/llm_finetune/devstral/devstral2_small_2512_squad.yaml>` | SFT — Devstral Small 2 24B on SQuAD |
| {download}`devstral2_small_2512_squad_peft.yaml <../../../examples/llm_finetune/devstral/devstral2_small_2512_squad_peft.yaml>` | LoRA — Devstral Small 2 24B on SQuAD |


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
automodel --nproc-per-node=8 examples/llm_finetune/devstral/devstral2_small_2512_squad.yaml
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
automodel --nproc-per-node=8 examples/llm_finetune/devstral/devstral2_small_2512_squad.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [mistralai/Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512)
- [mistralai/Devstral-Small-2-24B-Instruct-2512](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512)
