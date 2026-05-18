# Solar Pro

[Solar Pro](https://huggingface.co/upstage/solar-pro-preview-instruct) is an enterprise language model from Upstage, built on a depth up-scaling technique applied to Llama-based architectures.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `SolarForCausalLM` |
| **Parameters** | 22B |
| **HF Org** | [upstage](https://huggingface.co/upstage) |
:::

## Available Models

- **solar-pro-preview-instruct**

## Architecture

- `SolarForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Solar Pro Preview Instruct | [`upstage/solar-pro-preview-instruct`](https://huggingface.co/upstage/solar-pro-preview-instruct) |


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

- [upstage/solar-pro-preview-instruct](https://huggingface.co/upstage/solar-pro-preview-instruct)
