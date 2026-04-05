# ChatGLM

[ChatGLM](https://github.com/THUDM/ChatGLM-6B) is a bilingual (Chinese-English) conversational language model from Tsinghua University (THUDM). ChatGLM2 and ChatGLM3 extend the original with improved performance, longer context, and more efficient attention.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `ChatGLMModel` |
| **Parameters** | 6B |
| **HF Org** | [THUDM](https://huggingface.co/THUDM) |
:::

## Available Models

- **ChatGLM3-6B**
- **ChatGLM2-6B**

## Architecture

- `ChatGLMModel` / `ChatGLMForConditionalGeneration`

## Example HF Models

| Model | HF ID |
|---|---|
| ChatGLM3 6B | `THUDM/chatglm3-6b` |
| ChatGLM2 6B | `THUDM/chatglm2-6b` |


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

- [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b)
- [THUDM/chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b)
