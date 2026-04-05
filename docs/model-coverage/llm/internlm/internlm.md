# InternLM

[InternLM](https://github.com/InternLM/InternLM) is a bilingual (Chinese-English) language model series from Shanghai AI Laboratory, with versions 1, 2, and 3 each improving on reasoning, instruction following, and long-context capabilities.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `InternLMForCausalLM` / `InternLM2ForCausalLM` / `InternLM3ForCausalLM` |
| **Parameters** | 7B – 8B |
| **HF Org** | [internlm](https://huggingface.co/internlm) |
:::

## Available Models

- **InternLM3-8B-Instruct** (InternLM3)
- **InternLM2-7B**, **InternLM2-Chat-7B** (InternLM2)
- **InternLM-7B**, **InternLM-Chat-7B** (InternLM v1)

## Architectures

- `InternLMForCausalLM` — InternLM v1
- `InternLM2ForCausalLM` — InternLM2
- `InternLM3ForCausalLM` — InternLM3

## Example HF Models

| Model | HF ID |
|---|---|
| InternLM3 8B Instruct | `internlm/internlm3-8b-instruct` |
| InternLM2 7B | `internlm/internlm2-7b` |
| InternLM 7B | `internlm/internlm-7b` |


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

- [internlm/internlm3-8b-instruct](https://huggingface.co/internlm/internlm3-8b-instruct)
- [internlm/internlm2-7b](https://huggingface.co/internlm/internlm2-7b)
