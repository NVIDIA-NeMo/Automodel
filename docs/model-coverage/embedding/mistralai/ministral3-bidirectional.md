# Ministral3 (Bidirectional) for Embedding

NeMo AutoModel provides a bidirectional variant of [Mistral AI's Ministral3](https://mistral.ai/news/ministraux/) for embedding and dense retrieval tasks. Unlike the standard causal (left-to-right) Ministral3 used for text generation, this variant uses **bidirectional attention**, so each token can attend to both past and future tokens in the sequence, producing richer representations for semantic similarity and dense retrieval.

The bidirectional encoder can be loaded directly from text-only checkpoints (e.g. `mistralai/Ministral-3B-Instruct`) and also automatically extracts the language model from Ministral3 VLM checkpoints (e.g. `mistralai/Ministral-3-3B-Base-2512` or `mistralai/Ministral-3-3B-Instruct-2512`).

:::{card}
| | |
|---|---|
| **Tasks** | Embedding, Dense Retrieval |
| **Architecture** | `Ministral3BidirectionalModel` |
| **Parameters** | 3B |
| **HF Org** | [mistralai](https://huggingface.co/mistralai) |
:::

## Available Models

Any Ministral3 checkpoint can be loaded as a bidirectional backbone. The following configurations are tested:

- **Ministral-3-3B-Base-2512** — VLM checkpoint, language model is extracted automatically
- **Ministral-3-3B-Instruct-2512** — VLM checkpoint, language model is extracted automatically

## Embedding Models

The bidirectional bi-encoder path is used for embedding generation and dense retrieval.

| Architecture | Task | Auto Class | Description |
|---|---|---|---|
| `Ministral3BidirectionalModel` | Embedding | [`NeMoAutoModelBiEncoder`](https://github.com/NVIDIA-NeMo/Automodel/blob/8dc00dcb4a35c2413c52c6e7eb7ac8f1c24836aa/nemo_automodel/_transformers/auto_model.py#L991) | Bidirectional Ministral3 with mean pooling for dense embeddings |

## Pooling Strategies

The bi-encoder supports multiple pooling strategies to aggregate token representations into a single embedding vector:

| Strategy | Description |
|---|---|
| `avg` | Average of all token hidden states (default) |
| `cls` | First token hidden state |
| `last` | Last non-padding token hidden state |
| `weighted_avg` | Weighted average of token hidden states |

## Example HF Models

| Model | HF ID |
|---|---|
| Ministral-3 3B Base | [`mistralai/Ministral-3-3B-Base-2512`](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512) |
| Ministral-3 3B Instruct | [`mistralai/Ministral-3-3B-Instruct-2512`](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) |

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

**3. Run the recipe** from inside the repo (point any Llama bi-encoder recipe at a Ministral3 checkpoint, or write a recipe targeting `mistralai/Ministral-3-3B-Base-2512`):

```bash
torchrun --nproc-per-node=8 examples/retrieval/bi_encoder/finetune.py --config examples/retrieval/bi_encoder/llama3_2_1b.yaml
torchrun --nproc-per-node=8 examples/retrieval/bi_encoder/finetune.py --config examples/retrieval/bi_encoder/ministral3_3b_instruct.yaml

See the [Installation Guide](../../../guides/installation.md).

<!-- TODO: uncomment when finetune guide is published.
## Fine-Tuning

See the [Embedding and Reranking Fine-Tuning Guide](../../../guides/retrieval/finetune.md) for bi-encoder training instructions, including LoRA and PEFT configuration.
-->

## Hugging Face Model Cards

- [mistralai/Ministral-3-3B-Base-2512](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512)
- [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)
