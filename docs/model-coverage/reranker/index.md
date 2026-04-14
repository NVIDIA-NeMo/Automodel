(reranking-models)=

# Reranking Models

## Introduction

Reranking models use cross-encoders to score a query-document pair jointly. They are typically used after an embedding model has produced an initial candidate set. NeMo AutoModel supports optimized bidirectional Llama rerankers and falls back to HuggingFace `AutoModelForSequenceClassification` for other architectures.

For first-stage dense retrieval, see [Embedding Models](../embedding/index.md).

## Optimized Backbones (Bidirectional Attention)

| Owner | Model | Architecture | Wrapper Class | Tasks |
|---|---|---|---|---|
| Meta | [Llama (Bidirectional)](meta/llama-bidirectional.md) | `LlamaBidirectionalForSequenceClassification` | `NeMoAutoModelCrossEncoder` | Reranking |

## HuggingFace Auto Backbones

Any HuggingFace model loadable via `AutoModelForSequenceClassification` can be used as a reranking backbone. This fallback path uses the model's native attention -- no bidirectional conversion is applied.

## Supported Workflows

- **Fine-tuning (Cross-Encoder):** Cross-entropy training on query-document pairs to produce rerankers
- **LoRA/PEFT:** Parameter-efficient fine-tuning for reranking backbones

## Dataset

Retrieval fine-tuning requires query-document pairs: each example is a query paired with one positive document and one or more negative documents. Both inline JSONL and corpus ID-based JSON formats are supported. See the [Retrieval Dataset](../../guides/llm/retrieval-dataset.md) guide.

## Train Reranking Models

For a complete walkthrough of training configuration, model-specific settings, and launch commands, see the [Embedding and Reranking Fine-Tuning Guide](../../guides/retrieval/finetune.md).

```{toctree}
:hidden:

meta/llama-bidirectional
```
