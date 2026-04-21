# Llama (Bidirectional) for Embedding

NeMo AutoModel provides a bidirectional variant of [Meta's Llama](https://www.llama.com/) for embedding and dense retrieval tasks. Unlike the standard causal (left-to-right) Llama used for text generation, this variant uses **bidirectional attention**, so each token can attend to both past and future tokens in the sequence, producing richer representations for semantic similarity and dense retrieval.

For the cross-encoder variant, see [Llama (Bidirectional) for Reranking](../../reranker/meta/llama-bidirectional.md).

:::{card}
| | |
|---|---|
| **Tasks** | Embedding, Dense Retrieval |
| **Architecture** | `LlamaBidirectionalModel` |
| **Parameters** | 1B – 8B |
| **HF Org** | [meta-llama](https://huggingface.co/meta-llama) |
:::

## Available Models

Any Llama checkpoint can be loaded as a bidirectional backbone. The following configurations are tested:

- **Llama 3.2 1B** — fast iteration, fits on a single GPU
- **Llama 3.1 8B** — higher-quality embeddings for production use

## Embedding Models

The bidirectional bi-encoder path is used for embedding generation and dense retrieval.

| Architecture | Task | Auto Class | Description |
|---|---|---|---|
| `LlamaBidirectionalModel` | Embedding | [`NeMoAutoModelBiEncoder`](https://github.com/NVIDIA-NeMo/Automodel/blob/8dc00dcb4a35c2413c52c6e7eb7ac8f1c24836aa/nemo_automodel/_transformers/auto_model.py#L991) | Bidirectional Llama with mean pooling for dense embeddings |

## Pooling Strategies

The bi-encoder supports multiple pooling strategies to aggregate token representations into a single embedding vector:

| Strategy | Description |
|---|---|
| `avg` | Average of all token hidden states (default) |
| `cls` | First token hidden state |
| `last` | Last non-padding token hidden state |
| `weighted_avg` | Weighted average of token hidden states |
| `colbert` | No pooling — token-level embeddings (ColBERT-style) |

## Example HF Models

| Model | HF ID |
|---|---|
| Llama 3.2 1B | [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) |
| Llama 3.1 8B | [`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`llama3_2_1b.yaml <../../../../examples/retrieval/bi_encoder/llama3_2_1b.yaml>` | Bi-encoder — Llama 3.2 1B embedding model |

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

**3. Run the recipe** from inside the repo:

```bash
automodel examples/retrieval/bi_encoder/llama3_2_1b.yaml --nproc-per-node 8
```

:::{dropdown} Run with Docker
**1. Pull the container** and mount a checkpoint directory:

```bash
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v $(pwd)/checkpoints:/opt/Automodel/checkpoints \
  nvcr.io/nvidia/nemo-automodel:26.02.00
```

**2. Navigate** to the AutoModel directory (where the recipes are):

```bash
cd /opt/Automodel
```

**3. Run the recipe**:

```bash
automodel examples/retrieval/bi_encoder/llama3_2_1b.yaml --nproc-per-node 8
```
:::

See the [Installation Guide](../../../guides/installation.md).

<!-- TODO: uncomment when finetune guide is published.
## Fine-Tuning

See the [Embedding and Reranking Fine-Tuning Guide](../../../guides/retrieval/finetune.md) for bi-encoder training instructions, including LoRA and PEFT configuration.
-->

## Hugging Face Model Cards

- [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
