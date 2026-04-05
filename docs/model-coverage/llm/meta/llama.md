# Llama

[Meta's Llama](https://www.llama.com/) is a family of open-weight autoregressive language models built on the transformer decoder architecture. Key design choices include pre-normalization with RMSNorm, SwiGLU activations, and Rotary Positional Embeddings (RoPE). Llama 3+ models add Grouped Query Attention (GQA) for memory-efficient inference at larger scales.

:::{card}
| | |
|---|---|
| **Task** | Text Generation |
| **Architecture** | `LlamaForCausalLM` |
| **Parameters** | 1B – 405B |
| **HF Org** | [meta-llama](https://huggingface.co/meta-llama) |
:::

## Available Models

- **Llama 3.2**: 1B, 3B
- **Llama 3.1**: 8B, 70B, 405B (128K context)
- **Llama 3**: 8B, 70B
- **Llama 2**: 7B, 13B, 70B
- **LLaMA (v1)**: 7B, 13B, 30B, 65B
- **Yi** (01-ai): 6B, 34B — uses `LlamaForCausalLM`

## Architecture

- `LlamaForCausalLM`

## Example HF Models

| Model | HF ID |
|---|---|
| Llama 3.2 1B | `meta-llama/Llama-3.2-1B` |
| Llama 3.2 3B | `meta-llama/Llama-3.2-3B` |
| Llama 3.1 8B | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Llama 3.1 70B | `meta-llama/Meta-Llama-3.1-70B-Instruct` |
| Llama 3.1 405B | `meta-llama/Meta-Llama-3.1-405B-Instruct` |
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Llama 3 70B | `meta-llama/Meta-Llama-3-70B-Instruct` |
| Llama 2 70B | `meta-llama/Llama-2-70b-hf` |
| Yi 34B | `01-ai/Yi-34B` |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`llama3_2_1b_squad.yaml <../../../examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml>` | SFT — Llama 3.2 1B on SQuAD |
| {download}`llama_3_3_70b_instruct_squad.yaml <../../../examples/llm_finetune/llama3_3/llama_3_3_70b_instruct_squad.yaml>` | SFT — Llama 3.3 70B Instruct on SQuAD |


## Try with NeMo AutoModel

```bash
automodel --nproc-per-node=8 examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```

:::{dropdown} Run with Docker
Pull the NeMo AutoModel container and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```

Then inside the container:

```bash
automodel --nproc-per-node=8 examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml
```
:::

See the [Installation Guide](../../../guides/installation.md) and [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md).

## Fine-Tuning

See the [LLM Fine-Tuning Guide](../../../guides/llm/finetune.md) for full SFT and LoRA instructions.

## Hugging Face Model Cards

- [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
- [meta-llama/Meta-Llama-3.1-70B](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B)
- [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
