# Llama-Embed-Nemotron-8B

[Llama-Embed-Nemotron-8B](https://huggingface.co/nvidia/llama-embed-nemotron-8b) is NVIDIA's text embedding model for retrieval, semantic similarity, classification, and multilingual retrieval workloads. In NeMo AutoModel, it is reproduced with the bidirectional Llama bi-encoder backbone.

For architecture-level details such as bidirectional attention and pooling strategies, see [Llama (Bidirectional)](./llama-bidirectional.md).

:::{card}
| | |
|---|---|
| **Task** | Embedding, Dense Retrieval |
| **Architecture** | `LlamaBidirectionalModel` |
| **Parameters** | 8B |
| **HF Org** | [nvidia](https://huggingface.co/nvidia) |
:::

## Available Models

- **Llama-Embed-Nemotron-8B**

## Architecture

- `LlamaBidirectionalModel`

## Example HF Models

| Model | HF ID |
|---|---|
| Llama-Embed-Nemotron-8B | [`nvidia/llama-embed-nemotron-8b`](https://huggingface.co/nvidia/llama-embed-nemotron-8b) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`llama_embed_nemotron_8b.yaml <../../../../examples/retrieval/bi_encoder/llama_embed_nemotron_8b/llama_embed_nemotron_8b.yaml>` | Bi-encoder — reproduction recipe for Llama-Embed-Nemotron-8B |

## Try with NeMo AutoModel

**1. Install NeMo AutoModel**. Refer to the ([Installation Guide](../../../guides/installation.md)) for information:

```bash
uv pip install nemo-automodel
```

**2. Clone the repo** to get the example recipes:

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel
```

**3. Prepare the dataset** used by the reproduction recipe:

```bash
uv run python examples/retrieval/bi_encoder/llama_embed_nemotron_8b/data_preparation.py \
  --download-path ./embed_nemotron_dataset_v1
```

**4. Run the recipe** from inside the repo:

```bash
automodel examples/retrieval/bi_encoder/llama_embed_nemotron_8b/llama_embed_nemotron_8b.yaml --nproc-per-node 8
```

See the [Installation Guide](../../../guides/installation.md).

<!-- TODO: uncomment when finetune guide is published.
## Fine-Tuning

See the [Embedding and Reranking Fine-Tuning Guide](../../../guides/retrieval/finetune.md) for bi-encoder training instructions, including LoRA and PEFT configuration.
-->

## Hugging Face Model Cards

- [nvidia/llama-embed-nemotron-8b](https://huggingface.co/nvidia/llama-embed-nemotron-8b)
- [nvidia/embed-nemotron-dataset-v1](https://huggingface.co/datasets/nvidia/embed-nemotron-dataset-v1)
