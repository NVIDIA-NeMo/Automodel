# OLMo2

[OLMo2](https://allenai.org/olmo) is Allen AI's second-generation open language model with improved architecture and training, including RMSNorm and rotary position embeddings.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `OLMo2ForCausalLM` |
| **Parameters** | 1B – 13B |
| **HF Org** | [allenai](https://huggingface.co/allenai) |
:::

## Available Models

- **OLMo2-0425-1B-Instruct**
- **OLMo2-7B-1124**
- **OLMo2-13B-1124**

## Architecture

- `OLMo2ForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| OLMo2 7B | `allenai/OLMo2-7B-1124` |
| OLMo2 0425 1B Instruct | `allenai/OLMo2-0425-1B-Instruct` |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`olmo_2_0425_1b_instruct_squad.yaml <../../../examples/llm_finetune/olmo/olmo_2_0425_1b_instruct_squad.yaml>` | SFT — OLMo2 0425 1B Instruct on SQuAD |
| {download}`olmo_2_0425_1b_instruct_squad_peft.yaml <../../../examples/llm_finetune/olmo/olmo_2_0425_1b_instruct_squad_peft.yaml>` | LoRA — OLMo2 0425 1B Instruct on SQuAD |


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
automodel --nproc-per-node=8 examples/llm_finetune/olmo/olmo_2_0425_1b_instruct_squad.yaml
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
automodel --nproc-per-node=8 examples/llm_finetune/olmo/olmo_2_0425_1b_instruct_squad.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [allenai/OLMo2-7B-1124](https://huggingface.co/allenai/OLMo2-7B-1124)
