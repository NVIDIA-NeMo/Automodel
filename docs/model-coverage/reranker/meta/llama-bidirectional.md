# Llama (Bidirectional)

NeMo AutoModel provides a bidirectional variant of [Meta's Llama](https://www.llama.com/) for reranking tasks. Unlike the standard causal (left-to-right) Llama used for text generation, this variant uses **bidirectional attention** so the query and document can interact across the full sequence before a classification head produces a relevance score.

For the bi-encoder variant, see [Llama (Bidirectional) for Embedding](../../embedding/meta/llama-bidirectional.md).

:::{card}
| | |
|---|---|
| **Tasks** | Reranking |
| **Architecture** | `LlamaBidirectionalForSequenceClassification` |
| **Parameters** | 1B – 8B |
| **HF Org** | [meta-llama](https://huggingface.co/meta-llama) |
:::

## Available Models

Any Llama checkpoint can be loaded as a bidirectional reranking backbone. Tested configurations:

- **Llama 3.2 1B** — fast iteration, fits on a single GPU
- **Llama 3.1 8B** — higher-quality reranking for production use

## Reranking Models

The cross-encoder path is used for pairwise relevance scoring and reranking.

| Architecture | Task | Wrapper Class | Description |
|---|---|---|---|
| `LlamaBidirectionalForSequenceClassification` | Reranking | `NeMoAutoModelCrossEncoder` | Bidirectional Llama with classification head for relevance scoring |

## Example HF Models

| Model | HF ID |
|---|---|
| Llama 3.2 1B | [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B) |
| Llama 3.1 8B | [`meta-llama/Llama-3.1-8B`](https://huggingface.co/meta-llama/Llama-3.1-8B) |

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`llama3_2_1b.yaml <../../../../examples/retrieval/cross_encoder/llama3_2_1b.yaml>` | Cross-encoder — Llama 3.2 1B reranker |

## Try with NeMo AutoModel

**1. Install** ([full instructions](../../../guides/installation.md)):

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
automodel examples/retrieval/cross_encoder/llama3_2_1b.yaml --nproc-per-node 8
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
automodel examples/retrieval/cross_encoder/llama3_2_1b.yaml --nproc-per-node 8
```
:::

See the [Installation Guide](../../../guides/installation.md) and [Embedding and Reranking Fine-Tuning Guide](../../../guides/retrieval/finetune.md).

## Fine-Tuning

See the [Embedding and Reranking Fine-Tuning Guide](../../../guides/retrieval/finetune.md) for cross-encoder training instructions, including LoRA/PEFT configuration.

## Hugging Face Model Cards

- [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
