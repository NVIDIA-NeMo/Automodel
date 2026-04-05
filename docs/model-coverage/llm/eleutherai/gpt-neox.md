# GPT-NeoX / Pythia

[GPT-NeoX](https://github.com/EleutherAI/gpt-neox) is EleutherAI's large-scale language model architecture. The same `GPTNeoXForCausalLM` architecture is used by the Pythia scaling suite, OpenAssistant, Databricks Dolly, and StableLM models.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `GPTNeoXForCausalLM` |
| **Parameters** | 1B – 20B |
| **HF Org** | [EleutherAI](https://huggingface.co/EleutherAI) |
:::

## Available Models

- **GPT-NeoX-20B** (EleutherAI)
- **Pythia** suite: 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B (EleutherAI)
- **OA-SFT-Pythia-12B** (OpenAssistant)
- **Dolly-v2-12B** (Databricks)
- **StableLM-tuned-alpha-7B** (Stability AI)

## Architecture

- `GPTNeoXForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| GPT-NeoX 20B | `EleutherAI/gpt-neox-20b` |
| Pythia 12B | `EleutherAI/pythia-12b` |
| OpenAssistant SFT Pythia 12B | `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5` |
| Dolly v2 12B | `databricks/dolly-v2-12b` |
| StableLM tuned alpha 7B | `stabilityai/stablelm-tuned-alpha-7b` |


## Try with NeMo AutoModel

Install NeMo AutoModel and follow the fine-tuning guide to configure a recipe for this model.

```bash
pip install nemo-automodel
```

:::{dropdown} Run with Docker
```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Hugging Face Model Cards

- [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b)
- [EleutherAI/pythia-12b](https://huggingface.co/EleutherAI/pythia-12b)
