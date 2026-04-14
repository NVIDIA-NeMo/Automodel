(embedding-models)=

# Embedding Models

## Introduction

Embedding models convert text into dense vector representations for semantic search, dense retrieval, retrieval-augmented generation (RAG), and classification. NeMo AutoModel supports optimized bidirectional Llama bi-encoders and falls back to HuggingFace `AutoModel` for other encoder backbones.

For cross-encoder pairwise scoring, see [Reranking Models](../reranker/index.md).

Embedding models use bi-encoders to produce dense representations for queries and documents independently. They are the standard path for embedding generation and first-stage dense retrieval.

### Optimized Backbones (Bidirectional Attention)

| Owner | Model | Architecture | Wrapper Class | Tasks |
|---|---|---|---|---|
| Meta | [Llama (Bidirectional)](meta/llama-bidirectional.md) | `LlamaBidirectionalModel` | `NeMoAutoModelBiEncoder` | Embedding, Dense Retrieval |
| NVIDIA | [Llama-Embed-Nemotron-8B](nvidia/llama-embed-nemotron-8b.md) | `LlamaBidirectionalModel` | `NeMoAutoModelBiEncoder` | Embedding, Dense Retrieval |

### HuggingFace Auto Backbones

Any HuggingFace model loadable via `AutoModel` can be used as an embedding backbone. This fallback path uses the model's native attention — no bidirectional conversion is applied.

## Example Recipes

| Recipe | Description |
|---|---|
| {download}`llama3_2_1b.yaml <../../../examples/retrieval/bi_encoder/llama3_2_1b.yaml>` | Bi-encoder — Llama 3.2 1B embedding model |
| {download}`llama_embed_nemotron_8b.yaml <../../../examples/retrieval/bi_encoder/llama_embed_nemotron_8b/llama_embed_nemotron_8b.yaml>` | Bi-encoder — Llama-Embed-Nemotron-8B reproduction recipe |

## Supported Workflows

- **Fine-tuning (Bi-Encoder):** Contrastive learning on query-document pairs to produce embedding models
- **LoRA/PEFT:** Parameter-efficient fine-tuning for embedding backbones
- **ONNX Export:** Export trained embedding models for deployment

## Dataset

Retrieval fine-tuning requires query-document pairs: each example is a query paired with one positive document and one or more negative documents. Both inline JSONL and corpus ID-based JSON formats are supported. See the [Retrieval Dataset](../../guides/llm/retrieval-dataset.md) guide.

## Train Embedding Models

For a complete walkthrough of training configuration, model-specific settings, and launch commands, see the [Embedding and Reranking Fine-Tuning Guide](../../guides/retrieval/finetune.md).

```{toctree}
:hidden:

meta/llama-bidirectional
nvidia/llama-embed-nemotron-8b
```
