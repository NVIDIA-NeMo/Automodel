# OLMo

[OLMo](https://allenai.org/olmo) (Open Language Model) is Allen AI's fully open language model — open weights, open training data, and open training code. OLMo-1B and OLMo-7B are trained on Dolma.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `OLMoForCausalLM` |
| **Parameters** | 1B – 7B |
| **HF Org** | [allenai](https://huggingface.co/allenai) |
:::

## Available Models

- **OLMo-7B-hf**: 7B
- **OLMo-1B-hf**: 1B

## Architecture

- `OLMoForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| OLMo 1B | `allenai/OLMo-1B-hf` |
| OLMo 7B | `allenai/OLMo-7B-hf` |


## Try with NeMo AutoModel

Install NeMo AutoModel and follow the fine-tuning guide to configure a recipe for this model.

**1. Install** ([full instructions](../../../guides/installation.md)):

```bash
pip install nemo-automodel
```

**2. Clone the repo** to get example recipes you can adapt:

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
```

**3. Fine-tune** by adapting a base LLM recipe — override the model ID on the CLI:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
  --model.pretrained_model_name_or_path <MODEL_HF_ID>
```

Replace `<MODEL_HF_ID>` with the model ID from **Example HF Models** above.

:::{dropdown} Run with Docker
**1. Pull the container** and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```

**2.** The recipes are at `/opt/Automodel/examples/` — navigate there:

```bash
cd /opt/Automodel
```

**3. Fine-tune**:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
  --model.pretrained_model_name_or_path <MODEL_HF_ID>
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [allenai/OLMo-1B-hf](https://huggingface.co/allenai/OLMo-1B-hf)
- [allenai/OLMo-7B-hf](https://huggingface.co/allenai/OLMo-7B-hf)
